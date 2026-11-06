# 数据集泛化评估

`evaluate_dataset.py` 的读取、滑窗、频谱预处理、Keeper＋躺姿头脚锁和计分都在一个脚本内，不导入原仓库的 `run`、`data2datasets`、`preprocess` 或其他部署脚本。PTH 模式的模型源码及其模型内部依赖随打包文件提供；ONNX 模式直接执行计算图。

## 运行

```bash
python deploy/evaluate_dataset.py \
  --root /path/to/dataset --date 20261023 --group 033 \
  --checkpoint /path/to/experiment/checkpoint/last.pth \
  --config /path/to/experiment/config/config.yaml \
  --device cuda:0
```

`--checkpoint` 按文件后缀自动选择后端：`.pth` 使用 PyTorch，`.onnx` 使用 ONNX Runtime，不会同时加载两个模型。例如将上面的参数换成：

```bash
--checkpoint /path/to/model_dynamic_batch.onnx
```

ONNX 使用动态 batch：每个窗口中连续有效且当前点数合格的人员一次推理，batch 随人数变化；无人合格时不调用模型。PC 图输入为 `points`（或 `input`）和可选的 `mask`；只声明 xyz 等部分特征时按图声明的维度截取输入，不影响六列原始点云的质量统计。BIN 图输入为 `input`、`bbox`、`bbox_mask`。姿态输出命名为 `pose`，或提供唯一的姿态输出，形状为 `[B,T,P,J,3]`；PC 按人输入时 P=1。输入数据类型按图声明转换。

`--config` 始终指训练配置，用于保持预处理一致。PTH 模式的配置目录需有 `model_config.yaml`；优先使用同目录保存的同名模型 `.py`，缺少时使用本地 `models/<模型名>/<模型名>.py`，并严格核对权重形状。ONNX 不需要模型 `.py` 或 `model_config.yaml`。BIN 模式仍需 `radar_config.yaml`（或用 `--radar-config` 指定）。评估只打印结果，不创建结果目录或保存文件。`--max-frames 50` 可先检查少量帧；正式评估不加此参数。

PTH 推理遵循 `task.precision`（FP32、FP16 或 BF16，未配置时为 FP32）；ONNX 使用计算图声明的精度。不同权重、精度、关节格式及样本筛选条件的历史结果需要分别核对。

输入格式按训练配置 `task.input` 自动识别：

| 输入 | 组内目录 | 模型 |
| --- | --- | --- |
| PC | `dpct高位机/PC/*.npy` | 按人点云模型，如 P4Transformer、SetTransformer |
| BIN / FFT | `dpct高位机/Bin/*.bin` | ResNet3D；DXYZ 或 DXYZ+KXYZ |

PC 数据应为 `[N,6]`，列为 xyz、velocity、power、tag，单位米。`pointcloud_source: fftgen` 时读取已生成的 `fftGen/PC`；也可用 `--pc-directory fftGen/PC` 指定。尚未加入 BIN 在线生成 FFTGen 点云。

同目录的 `frames.pcpack`、`frames.binpack`、`frames.gtpack` 加 `frames_index.npz` 也可直接读取。没有隐式 SSD/NAS 回退；`--root` 指向要评估的实际目录。

```text
ROOT/日期/
  calib/extrinsic_img_to_radar_high.npz
  calib/extrinsic_img_to_radar_low.npz
  data_collection/group_XXX/
    dpct高位机/PC/
    dpct高位机/Bin/              # BIN 模式
    camera results/smoothed 3D/  # COCO17 或 mocap21 的 pkl
```

GT 和质量点云按时间戳就近匹配，默认最大偏差 100 ms，可用 `--max-sync-ms` 调整。BIN 的质量判断使用就近的 PC 原始帧，缺少同步 PC 时跳过当前模型推理，Keeper 可沿用已有可信骨架；没有可用 Keeper 输出时不计分。数据源仅被读取，不写对齐缓存。

评估默认检查 GT 骨架合理性和时间连续性，COCO17 先转换为 MOCAP21：

- 四肢任一骨段 >0.85 m、肩宽 >0.8 m、髋宽 >0.7 m、同侧肩髋距离 >1 m，标记 `long_bone`；八段四肢中至少四段 <0.05 m，标记 `collapsed_bones`。这是训练侧采用的宽松结构阈值，不按站、坐、躺姿角度判错。
- Inf 标记 `invalid_numeric`；缺少同步 GT、无人或所有人员含 NaN 时标记 `missing_gt`。部分人员含 NaN 时，仅使对应人员的窗口无效，保留其他健康人员。
- 相邻匹配 GT 时间必须严格递增，重复使用同一 GT 也标记 `invalid_time`。输入或 GT 的相邻间隔超过 `--max-gap-ms`（默认 500 ms）标记 `time_gap`，可按实际采样周期调整。
- 同槽位 pelvis 位移 >max(0.35 m, 4 m/s × GT 时间差)，且十二个四肢关节位移中位数 >max(0.4 m, 4 m/s × GT 时间差)，标记 `pose_jump`。沿用训练侧跳变阈值，MOCAP21 使用实际四肢关节，避免插值脊柱点改变中位数权重。

上述异常清空推理窗口、积累点云、BIN/KXYZ 历史及 Keeper。结构损坏或缺失的当前帧不进入窗口；时间异常或跳变时，以当前有效骨架开始新窗口。历史重新就绪后恢复推理，不会用断点前的 Keeper 填补新窗口。这些检查针对 GT，不用预测误差筛选样本；几何检查无法识别所有缓慢换人或整体标定错误。

## 推理与计分口径

- 使用 **keeper_last**：步长固定为 1，每个完整窗口仅取最后一帧，经过 Keeper＋躺姿头脚锁后计分。沿用配置的 `T`、`acc_frame`、每人点数和输入预处理设置。按原始 GT 包围盒向外扩展 0.3 m 裁剪；确定性等距采样或重复本帧点，不随机采样。
- 窗口中同一 GT 人员槽位连续有效才输出最后一帧；没有可靠 GT 无法进行 GT 框裁剪。开头不足 T 帧、缺失 GT 等原因写入 `frame_reason`。KXYZ 另需六帧真实历史，不能以未来帧或补零代替。
- 质量统计使用当前未积累点云的高位坐标 GT 框：动态点 tag=1 至少 20 个，总点数至少 30 个。任一门槛不满足就跳过该人员当前窗口的模型推理，raw 留空（NaN）；同帧其他合格人员仍推理。无人合格时不调用模型，`frame_reason=low_point_count` 表示帧内存在因低点数跳过的人员。
- 窗口按真实时间继续滑动，质量恢复且窗口就绪后恢复推理。积累点数不替代当前帧点数；BIN 仍继续频谱预处理以保留后续窗口及 KXYZ 的真实历史。低点数时使用 Keeper 的维持骨架及 kept 状态，没有可信骨架时标记 excluded。有效维持结果仍按 keeper_last 参与计分。
- 后处理只使用 **Keeper＋躺姿头脚锁**，不运行 Guard，不使用新增双方向/80%动点确认。普通更新不限制位置和形变；肩髋同时反转或头脚轴反转超过 `max(100°, 360°/s × dt)` 时，需要连续三帧候选满足根位移 ≤1×dt 米、去平移平均形变 ≤2/3×dt 米、头脚轴变化 ≤30°。未确认持续保持，不回填历史。
- 首次接受有效躺姿时固定头脚向量；仍为躺姿但与固定轴夹角 >90° 或轴长度 <0.4 m，拒绝更新，三帧确认不能绕过。连续三帧站/坐蹲、相邻头脚方向 ≤30°，且该结果被采纳后解锁；不要求位置静止。缺 raw、低点数、未知姿态打断解锁计数但不解除锁。人员消失后重新出现重置状态。
- 按 MOCAP21 的 pelvis(0)、head(6)、肩(8/12)、髋(15/18)、踝(17/20) 计算。躺/站/坐蹲判定在高位到低位标定后的重力坐标系完成，规则与原 actionprocess 对应，头部使用 MOCAP head 节点；三帧形变按全部 21 点计算，不能假定与 COCO17 的数值逐帧完全一致。
- `accepted` 为直接接受，`confirmed` 为三帧确认后仅更新当前帧，`kept` 为保持上一可信骨架，`excluded` 为没有可用 Keeper 输出；没有可信历史时不会伪造骨架。
- 只统计 GT pelvis 转到低位坐标后落在椭圆内的样本：中心 `(2,0)` m，半轴 `(1.6,2.4)` m，包含边界。
- **最终计分只使用 Keeper 坐标，包含 accepted、confirmed、kept；excluded/absent 不计分。** 有限 GT、有限 Keeper 输出且位于椭圆内才计分；即使当前没有 raw，只要 Keeper 有有效维持输出也计分。按 GT 裁剪人员 ID 一一配对，所有三项指标使用同一人帧集合。
- 根据模型实际输出的关节数自动识别，所有骨架统一为仓库定义的 mocap21，不依赖配置中的 pose_format 标签。17 点 COCO 按原映射插值转换，21 点 MOCAP 保留原有关节坐标；其他点数明确报错。COCO17 转换时，其中脊柱等点由插值得到，并非独立标注。
- 全身取 21 点；躯干取 `pelvis, spine1, spine2, spine3, thorax, neck`（索引 0–5）；四肢取双侧肩、肘、腕、髋、膝、踝（索引 8、9、10、12、13、14、15、16、17、18、19、20，共 12 点）。按人帧等权平均，不单独输出头部误差；三项表格指标单位 **mm**，没有合格样本为 `N/A`。
- 人员沿用 GT 数组槽位，不执行检测或身份跟踪；明显跳变会中断窗口，但相近人员跨帧换槽仍可能影响窗口和 Keeper。

## 输出

标准输出只打印一行，不含表头，以制表符分隔，可直接粘入表格；进度写入标准错误：

```text
日期    group_XXX    全身误差_mm    躯干误差_mm    四肢误差_mm
```

评估不保存 `predictions.npz` 或其他结果文件，不支持 `--output` 参数。已有结果文件不受影响。骨架统一为 mocap21，与历史 COCO17 日志的数值需分别核对。

## 打包传递

```bash
python deploy/evaluate_dataset.py \
  --checkpoint /path/to/experiment/checkpoint/last.pth \
  --config /path/to/experiment/config/config.yaml \
  --bundle /path/to/evaluator.zip
```

ZIP 内包含评估脚本、所选模型、所需配置、说明和依赖列表，不包含数据集。PTH 包额外包含模型源码；ONNX 包只带 ONNX 模型，不带 PTH 和模型源码。解压后安装依赖，在解压目录按 `RUN.txt` 运行即可，不需要原仓库。已有 ZIP 不覆盖。

基础环境：Python ≥3.10、NumPy、PyYAML、PyTorch。ONNX 模式额外安装 `onnxruntime`；CUDA 推理改为安装 `onnxruntime-gpu`，并指定 `--device cuda:0`（按 [ONNX Runtime 官方 API](https://onnxruntime.ai/docs/api/python/api_summary.html) 选择执行后端）。ONNX 不需要 einops 或 pointnet2 扩展。PTH 模式部分模型需要 einops；P4Transformer/PCT 等使用 pointnet2 扩展的模型，还需要与目标 PyTorch/CUDA 匹配的 `pointnet2_ops`；不打包机器相关的已编译 CUDA 二进制。脚本及模型代码、pth、GT pickle 应来自可信来源。
