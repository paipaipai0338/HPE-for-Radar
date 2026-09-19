# 行为识别接入接口

`BehaviorRecognizer`接收检测和跟踪模块输出的高位机三维框，在内部完成点云关联、坐标转换、
多人重叠点归属、行为模型推理和状态后处理。

检测模块必须为同一个人提供稳定的`track_id`。三维框格式为高位机坐标系下的
`[xmin, ymin, zmin, xmax, ymax, zmax]`，点云格式为`N x 6`，与训练数据一致。
点云列依次为`x, y, z, Doppler, SNR, TLV`；当前模型不使用SNR，但后处理需要TLV区分动点和微动点。

```python
import numpy as np

from Rec import BehaviorRecognizer, PersonBox


recognizer = BehaviorRecognizer(
    checkpoint="/path/to/best.pt",
    low_extrinsic="/path/to/extrinsic_img_to_radar_low.npz",
    high_extrinsic="/path/to/extrinsic_img_to_radar_high.npz",
    device="cuda:0",
)

# 每个雷达帧调用一次，timestamp单位为秒且必须递增。
results = recognizer.update(
    high_points,  # np.ndarray, shape [N, 6]
    [
        PersonBox(track_id=3, box=np.asarray([xmin, ymin, zmin, xmax, ymax, zmax])),
    ],
    timestamp,
)

for result in results:
    print(result.track_id, result.label, result.confidence)
```

现有检测跟踪结果可直接转换：

```python
detections = [PersonBox(person.identifier, person.box) for person in tracked_people]
```

窗口积累完成前`label`为`"waiting"`。窗口完成后，`label`为`stand`、`sit_squat`、`lie`或
`other`；该结果已经过动态/静态门控和因果状态后处理。`point_count`是该人员最终关联的点数，
`holding`表示当前结果是否来自静态状态维持。

目标永久离开或上游重新分配ID时，可调用`recognizer.reset(track_id)`；切换数据流时调用
`recognizer.reset()`。

该目录是现有工程的简洁接入层，依赖项目中的`src/common`、`src/pointcloud`和`src/temporal`。
