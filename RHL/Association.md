# Detection 框后的点云关联

本文记录 `src/infer_pipeline_visualize.py` 中，检测网络得到高位机框后，生成单人行为识别点云的处理流程。

```text
高位机框内取种子点
        ↓
框角点转换到低位机坐标系
        ↓
低位机点云受限扩展
        ↓
多人重叠点归属
        ↓
行为模型
```

## 1. 坐标和输入

检测网络使用高位机原始点云。每帧点云通常为 `N x 6`：

```text
[x, y, z, doppler, snr, tlv]
```

在 [infer_pipeline_visualize.py](../../src/infer_pipeline_visualize.py) 中，`load_point_frames()` 同时保留高位机点云，并将坐标转换到低位机：

```python
high = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
finite = np.isfinite(high[:, :4]).all(axis=1)
finite &= np.isfinite(high[:, POINT_STATE_COLUMN]).all(axis=0)
high = high[finite]

low = high.copy()
low[:, :3] = transform_points_between_radars(
    high[:, :3], high_extrinsic, low_extrinsic
)
```

这里的 `high_points` 用于检测框取种子点，`points`（低位机坐标）用于后续扩展和行为模型。

## 2. 高位机框内取种子点

检测网络入口位于 [src/detection/inference.py](../../src/detection/inference.py) 的 `detect_people()`：

```python
detections = detect_people(
    detector,
    processor,
    high_points,
    device,
    score_threshold=DETECTION_SCORE_THRESHOLD,
    box_padding=DETECTION_BOX_PADDING,
    nms_iou_threshold=DETECTION_NMS_IOU_THRESHOLD,
)
```

检测结果的 `Detection.box` 是高位机坐标系下的轴对齐框，格式为：

```text
[xmin, ymin, zmin, xmax, ymax, zmax]
```

跟踪器先对检测框进行位置和生命周期更新：

```python
tracked_people = tracker.update(detections, high_points, timestamp)
```

随后使用最终的跟踪框重新取种子点，而不是继续使用检测阶段保存的点索引：

```python
seeds = [
    np.flatnonzero(
        np.all(
            (high_points[:, :3] >= person.box[:3])
            & (high_points[:, :3] <= person.box[3:]),
            axis=1,
        )
    )
    for person in tracked_people
]
```

种子点必须位于高位机跟踪框内部。此时仍未做扩展，也未将不同人员的点进行分离。

## 3. 框角点转换到低位机

扩展使用低位机坐标，因此先将每个高位机框的 8 个角点转换到低位机。

调用位置：

```python
boxes = [
    transform_box_corners(person.box, low_extrinsic, high_extrinsic)
    for person in tracked_people
]
```

`transform_box_corners()` 先构造轴对齐框的 8 个角点，再执行高位机到低位机的外参变换：

```python
def transform_box_corners(box, low_extrinsic, high_extrinsic):
    corners = box_corners(box)
    return transform_points_between_radars(
        corners,
        high_extrinsic,
        low_extrinsic,
    ).astype(np.float32)
```

转换后的 `boxes` 可能不再与低位机坐标轴对齐，但后续扩展函数会使用其各坐标的最小值和最大值构造限制范围。

## 4. 低位机点云受限扩展

扩展调用位于 [infer_pipeline_visualize.py](../../src/infer_pipeline_visualize.py)：

```python
candidates = [
    expand_box_seed_indices(
        points,
        seed,
        corners,
        PERSON_XY_EXPANSION_RADIUS,
        PERSON_Z_EXPANSION_RADIUS,
    )
    for seed, corners in zip(seeds, boxes, strict=True)
]
```

核心实现位于 [src/pointcloud/utils/pointcloud_io.py](../../src/pointcloud/utils/pointcloud_io.py) 的 `expand_box_seed_indices()`。

当前参数为：

```python
PERSON_XY_EXPANSION_RADIUS = 0.20
PERSON_Z_EXPANSION_RADIUS = 0.25
```

扩展逻辑如下：

1. 取转换后框角点的逐轴最小值和最大值；
2. 只保留 XY 投影位于框投影范围内的低位机点；
3. Z 方向允许在框上下边界外扩 `z_radius`；
4. 对每个候选点，检查它是否距离任一种子点满足：

   ```text
   XY 平面距离 ≤ xy_radius
   |Z 方向距离| ≤ z_radius
   ```
5. 满足条件的点作为该人员的候选点。

对应代码：

```python
def expand_box_seed_indices(
    points: np.ndarray,
    seed_indices: np.ndarray,
    box_corners: np.ndarray,
    xy_radius: float,
    z_radius: float,
) -> np.ndarray:
    """Add points directly neighboring box seeds within limited XY and Z bounds."""
    raw_points = np.asarray(points)
    seeds = np.asarray(seed_indices, dtype=int)
    corners = np.asarray(box_corners, dtype=np.float64)
    if raw_points.ndim != 2 or raw_points.shape[1] < 3:
        raise ValueError("points must have shape N x M with M >= 3")
    xyz = raw_points[:, :3]
    if corners.shape != (8, 3):
        raise ValueError("box_corners must have shape 8 x 3")
    if min(xy_radius, z_radius) <= 0:
        raise ValueError("xy_radius and z_radius must be positive")
    if not len(seeds):
        return seeds
    if seeds.min() < 0 or seeds.max() >= len(xyz):
        raise ValueError("seed_indices are outside points")

    lower, upper = corners.min(axis=0), corners.max(axis=0)
    candidate_mask = np.all((xyz[:, :2] >= lower[:2]) & (xyz[:, :2] <= upper[:2]), axis=1)
    candidate_mask &= (xyz[:, 2] >= lower[2] - z_radius) & (xyz[:, 2] <= upper[2] + z_radius)
    candidate_mask[seeds] = True
    candidate_indices = np.flatnonzero(candidate_mask)
    offsets = xyz[candidate_indices, None, :] - xyz[seeds, :]
    near_seed = (
        (np.linalg.norm(offsets[..., :2], axis=-1) <= xy_radius)
        & (np.abs(offsets[..., 2]) <= z_radius)
    ).any(axis=1)
    return candidate_indices[near_seed]
```

这一步使用低位机坐标下的全场景点云，但不会把框外任意远处的点加入进来。

## 5. 多人重叠点归属

多人框或扩展区域重叠时，同一个点可能同时出现在多个候选集合中。调用：

```python
assignments = resolve_person_point_overlap(
    points,
    seeds,
    candidates,
    np.asarray([box.mean(axis=0) for box in boxes]),
    PERSON_OVERLAP_MARGIN,
)
```

实现位于 `pointcloud_io.py` 的 `resolve_person_point_overlap()`。

```python
PERSON_OVERLAP_MARGIN = 0.10

def resolve_person_point_overlap(
    points: np.ndarray,
    seeds: list[np.ndarray],
    candidates: list[np.ndarray],
    centers: np.ndarray,
    ambiguity_margin: float = 0.10,
) -> list[np.ndarray]:
    """Resolve shared candidates in low-radar XY; preserve unambiguous box seeds."""
    if len(candidates) < 2:
        return candidates
    if len(seeds) != len(candidates) or np.asarray(centers).shape != (len(seeds), 3):
        raise ValueError("seeds, candidates and centers must describe the same people")
    if ambiguity_margin < 0:
        raise ValueError("ambiguity_margin must be non-negative")
    claims = np.zeros((len(candidates), len(points)), dtype=bool)
    seed_claims = np.zeros_like(claims)
    for row, (seed, candidate) in enumerate(zip(seeds, candidates, strict=True)):
        claims[row, candidate] = True
        seed_claims[row, seed] = True
    shared = claims.sum(axis=0) > 1
    if not shared.any():
        return candidates
    seed_counts = seed_claims.sum(axis=0)
    references = np.asarray(centers, dtype=float)[:, :2].copy()
    for row in range(len(seeds)):
        exclusive = seed_claims[row] & (seed_counts == 1)
        # Very sparse exclusive points may be a limb or an outlier.
        if exclusive.sum() >= 5:
            references[row] = np.median(points[exclusive, :2], axis=0)
    for index in np.flatnonzero(shared):
        owners = np.flatnonzero(claims[:, index])
        claims[:, index] = False
        if seed_counts[index] == 1:
            claims[np.flatnonzero(seed_claims[:, index])[0], index] = True
            continue
        distances = np.linalg.norm(references[owners] - points[index, :2], axis=1)
        order = np.argsort(distances)
        if distances[order[1]] - distances[order[0]] > ambiguity_margin:
            claims[owners[order[0]], index] = True
    return [np.flatnonzero(mask) for mask in claims]
```

处理规则：

1. 初始种子点优先级最高；
2. 只被一个人声明的种子点保留给该人；
3. 对共享候选点，计算其到各人员参考中心的 XY 距离；
4. 如果最近人员明显更近，则分配给最近人员；
5. 最近和次近距离之差不超过 `ambiguity_margin` 时，放弃该共享点，避免串入错误目标。

参考中心默认为转换后框的中心。如果某个人拥有至少 5 个排他的种子点，则改用这些排他种子点的 XY 中位数作为参考中心。

## 6. 送入行为模型

每个人最终获得一组不重复的点索引：

```python
for person, corners, seed, behavior_indices in zip(
    tracked_people,
    boxes,
    seeds,
    assignments,
    strict=True,
):
    person_points = points[behavior_indices]
```

随后进入该人员自己的时序缓存和行为模型：

```python
label, raw_label, confidence = classify_available_points(
    behavior_model,
    behavior_config,
    track,
    points[behavior_indices],
    device,
)
```

每个跟踪 ID 独立维护：

- 时序点云窗口；
- 行为预测结果；
- 静态状态维持；
- 后处理状态。

因此，多人情况下的处理单位是“每个人一个点云序列”，不是把多人点云合并后做一次行为分类。

## 7. 相关参数位置

参数集中在 `src/infer_pipeline_visualize.py` 顶部：

```python
DETECTION_BOX_PADDING = (0.10, 0.10, 0.10)
DETECTION_NMS_IOU_THRESHOLD = 0.25
PERSON_XY_EXPANSION_RADIUS = 0.20
PERSON_Z_EXPANSION_RADIUS = 0.25
PERSON_OVERLAP_MARGIN = 0.10
```

调整影响：

- `DETECTION_BOX_PADDING`：改变检测框和初始种子点范围；
- `PERSON_XY_EXPANSION_RADIUS`：改变水平邻域扩展强度；
- `PERSON_Z_EXPANSION_RADIUS`：改变竖直方向扩展范围；
- `PERSON_OVERLAP_MARGIN`：改变多人共享点的保守程度。

训练数据中的姿态框关联也复用了 `expand_box_seed_indices()`，但初始框来自姿态而不是 detection 网络，代码位于 `src/pointcloud/data/dataset.py` 的 `_associated_points()`。
