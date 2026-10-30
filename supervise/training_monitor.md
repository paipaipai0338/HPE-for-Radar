# 训练监控

最后刷新：2026-10-30 14:30:01 +08:00；刷新间隔：10 分钟；采集状态：成功。任务 / 采集提示：1 项，见下文。

下次计划刷新：2026-10-30 14:40:00 +08:00。超过 20 分钟未更新请检查 [health.json](</home/pai/Huawei/supervise/health.json>) 和 [monitor.log](</home/pai/Huawei/supervise/monitor.log>)。

范围：tasks.json 登记的项目任务；GPU 表列出全机计算占用。MPJPE 统一为 cm；不同骨架与数据划分的指标不宜直接横比。

## 当前任务

| 任务 | 状态 | GPU | 已完成 epoch | 最近验证 MPJPE | 预计结束时间 |
| --- | --- | --- | --- | --- | --- |
| P4Transformer bbox扰动，GPU0最高优先级微调50轮 | 运行中 | 0 | 340/350 | 14.844 cm | 2026-10-30 20:40:03 +08:00 |
| P4Transformer MOCAP-21，步长1微调50轮 | 运行中 | 1 | 324/350 | 13.347 cm | 2026-10-31 13:09:24 +08:00 |

预计结束时间为北京时间的训练结束估计，不含后续评估；最多取最近 11 条完整 epoch 日志，有多少用多少。只有一条时按进程启动至该轮完成的耗时估算（含初始化）；无有效耗时数据或正在评估时显示“暂无法估算”。

## 排队与后续任务

| 任务 | 状态 | GPU | 启动条件 / 内容 |
| --- | --- | --- | --- |
| ResNet3D 最佳版续训25–48轮，GPU0排队首位 | 排队中 | 0 | 当前GPU0 bbox微调保持运行；其结束且锁释放后优先执行，先于多模型队列；GPU0三次空闲检查后开始。 |
| P4Transformer xyz，步长1微调50轮 | 排队中 | 1 | 等待GPU1当前MOCAP微调退出，再取得GPU1共享锁；保留原50轮微调与--start-now策略。 |
| P4Transformer xyzv，步长1微调50轮 | 排队中 | 1 | 等待前序xyz 50轮微调退出，再取得GPU1共享锁；按用户要求跳过整卡空闲检查，微调260–309轮。 |
| P4Transformer xyz 多模型对比，微调 51–100 epoch | 排队中 | 0 | 第1/5项：等待当前bbox微调及最高优先级ResNet3D第25–48轮结束、旧暂停控制进程释放锁，再通过GPU0三次空闲检查。 |
| PointNet xyz 多模型对比，微调 51–100 epoch | 排队中 | 0 | 同一 GPU0 串行队列第 2/5 项；等待 P4Transformer 本阶段训练完成。 |
| SetTransformer xyz 多模型对比，微调 51–100 epoch | 排队中 | 0 | 同一 GPU0 串行队列第 3/5 项；等待 PointNet 本阶段训练完成。 |
| VoxelNeXt xyz 多模型对比，微调 51–100 epoch | 排队中 | 0 | 同一 GPU0 串行队列第 4/5 项；等待 SetTransformer 本阶段训练完成。 |
| ConvNeXtV2 xyz 多模型对比，从头训练 100 epoch | 排队中 | 0 | 同一 GPU0 串行队列第 5/5 项；等待 VoxelNeXt 本阶段训练完成。 |

顺序来自现有任务和启动条件；本监控不负责启动、重试或调整训练。

## GPU 占用

| GPU | 型号 | 利用率 | 显存 MiB（已用 / 总计） |
| --- | --- | --- | --- |
| 0 | NVIDIA GeForce RTX 5090 | 51% | 11773 / 32607 |
| 1 | NVIDIA GeForce RTX 5090 | 64% | 16049 / 32607 |

| GPU | PID | 用户 | 归属 | 命令 |
| --- | --- | --- | --- | --- |
| 0 | 252243 | pai | 已登记任务 | /home/pai/miniconda3/envs/pytorch/bin/python -B -u /home/pai/Huawei/experiments/P4Transformer/priority_bbox_stride1_50_20261029_103145/launch.py --worker |
| 0 | 1075450 | orange | 未登记占用（不推断实验配置） | python src/pointcloud/train_refiner.py |
| 1 | 246250 | pai | 已登记任务 | /home/pai/miniconda3/envs/pytorch/bin/python -B -u /home/pai/Huawei/experiments/P4Transformer_mocap_300/queue_stride1_100_20261028_213911/launch.py --worker |

## 异常与待处理

- ConvNeXtV2 xyz-only，50 epoch：OutOfMemoryError('CUDA out of memory. Tried to allocate 3.75 GiB. GPU 0 has a total capacity of 31.36 GiB of which 2.09 GiB is free. Including non-PyTorch memory, this process has 29.24 GiB memory in use. Of the allocated memory 26.22 GiB is allocated by PyTorch, and 2.40 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)')

## 已完成、失败或待核实的登记任务

| 任务 | 状态 | GPU | 已完成 epoch | 最近验证 MPJPE |
| --- | --- | --- | --- | --- |
| P4Transformer MOCAP-21，300 epoch | 已完成 | 0 | 300/300 | 14.209 cm |
| P4Transformer T=3 时间窗口消融 | 已完成 | 1 | 300/300 | 14.618 cm |
| P4Transformer bbox 扰动对比 | 已完成 | 0 | 300/300 | 15.531 cm |
| P4Transformer T=1 时间窗口消融 | 已完成 | 1 | 300/300 | 15.136 cm |
| ConvNeXtV2 xyz-only，50 epoch | 已失败 | 0 | 0/50 | 未记录 |

## 任务详情与证据

### ResNet3D 最佳版续训25–48轮，GPU0排队首位

从20261023_065632第24轮last恢复权重、AdamW及指标历史，在当前数据清单（请求训练504组/验证221组，含俯卧）继续到48轮；COCO-17、DXYZ+KXYZ、GT ROI、T=1、步长1、batch16、BF16、每轮8000批；平移±10cm、镜像50%、功率增强、骨长约束0.2；lr=0.00004、余弦24轮、无预热。新验证清单独立统计best，历史实验保留。

- 状态：排队中；阶段：waiting_predecessor；已完成 24/48 epoch。
- 配置：[config.yaml](</home/pai/Huawei/experiments/ResNet3D/20261030_100710/config/config.yaml>)；状态依据：[status.json](</home/pai/Huawei/experiments/ResNet3D/20261030_100710/continuation/status.json>)。
- PID：1007656；用户：pai；进程开始：2026-10-30 10:08:31 +08:00。
- 控制台：尚未生成。
- 等待日志：[queue.log](</home/pai/Huawei/experiments/ResNet3D/20261030_100710/continuation/queue.log>)。
- 最近证据更新时间：2026-10-30 10:08:31 +08:00。
- 实验目录：[20261030_100710](</home/pai/Huawei/experiments/ResNet3D/20261030_100710>)。

### P4Transformer bbox扰动，GPU0最高优先级微调50轮

从20261028_161946第300轮last恢复，T=16、训练/验证步长1，微调301–350轮；lr=0.0003、无预热、余弦50轮、batch=64、BF16，保留每帧bbox扰动和跟踪框增强。日志、权重和配置续写原实验目录；300轮结果备份于backup_before_stride1_50。原GPU0多模型队列整体内存暂停，本任务退出后自动原地恢复。

- 状态：运行中；阶段：training；已完成 340/350 epoch。
- 配置：[resume_stride1_350.yaml](</home/pai/Huawei/experiments/P4Transformer/20261028_161946/config/resume_stride1_350.yaml>)；状态依据：[status.json](</home/pai/Huawei/experiments/P4Transformer/20261028_161946/finetune_stride1_50/status.json>)。
- PID：252243；用户：pai；进程开始：2026-10-29 10:35:02 +08:00。
- 最近完整 epoch：340（2026-10-30 14:01:33 +08:00）；训练 MPJPE：6.715 cm；验证 MPJPE：14.844 cm；验证 loss：0.14844395059781307。
- 已记录的最佳验证 MPJPE：14.813 cm（epoch 335）。
- 预计训练结束：2026-10-30 20:40:03 +08:00（最近完整 epoch 耗时中位数，仅供参考，不含后续评估）。
- 训练日志：[log.txt](</home/pai/Huawei/experiments/P4Transformer/20261028_161946/log/log.txt>)。
- 控制台：[finetune_console.log](</home/pai/Huawei/experiments/P4Transformer/20261028_161946/log/finetune_console.log>)。
- 最近证据更新时间：2026-10-30 14:30:01 +08:00。
- 实验目录：[20261028_161946](</home/pai/Huawei/experiments/P4Transformer/20261028_161946>)。
- last.pth：[last.pth](</home/pai/Huawei/experiments/P4Transformer/20261028_161946/checkpoint/last.pth>)（更新于 2026-10-30 14:01:33 +08:00）。
- best.pth：[best.pth](</home/pai/Huawei/experiments/P4Transformer/20261028_161946/checkpoint/best.pth>)（更新于 2026-10-30 10:47:34 +08:00）。

### P4Transformer MOCAP-21，300 epoch

保持原训练条件，将 COCO-17 改为 MOCAP-21，训练后比较骨架插值精度。

- 状态：已完成；阶段：complete；已完成 300/300 epoch。
- 配置：[config.yaml](</home/pai/Huawei/experiments/P4Transformer_mocap_300/queue_20261027/config.yaml>)；状态依据：[status.json](</home/pai/Huawei/experiments/P4Transformer_mocap_300/queue_20261027/status.json>)。
- 控制台：[console.log](</home/pai/Huawei/experiments/P4Transformer_mocap_300/queue_20261027/console.log>)。
- 最近证据更新时间：2026-10-28 16:18:37 +08:00。
- 实验目录：[20261028_042133](</home/pai/Huawei/experiments/P4Transformer_mocap_300/20261028_042133>)。

### P4Transformer T=3 时间窗口消融

与已完成的 T=1 及原始基线比较，观察时间窗口长度对姿态精度的影响。

- 状态：已完成；阶段：complete；已完成 300/300 epoch。
- 配置：[t3_effective.yaml](</home/pai/Huawei/@codex/p4_window_ablation_20261026/t3_effective.yaml>)；状态依据：[t3_job.json](</home/pai/Huawei/@codex/p4_window_ablation_20261026/t3_job.json>)。
- 最近完整 epoch：300（2026-10-29 03:48:21 +08:00）；训练 MPJPE：6.074 cm；验证 MPJPE：14.618 cm；验证 loss：0.14617976658079626。
- 已记录的最佳验证 MPJPE：14.551 cm（epoch 213）。
- 训练日志：[log.txt](</home/pai/Huawei/experiments/P4Transformer_T3_300/20261028_063154/log/log.txt>)。
- 控制台：[t3.log](</home/pai/Huawei/@codex/p4_window_ablation_20261026/t3.log>)。
- 最近证据更新时间：2026-10-29 03:48:22 +08:00。
- 实验目录：[20261028_063154](</home/pai/Huawei/experiments/P4Transformer_T3_300/20261028_063154>)。
- last.pth：[last.pth](</home/pai/Huawei/experiments/P4Transformer_T3_300/20261028_063154/checkpoint/last.pth>)（更新于 2026-10-29 03:48:21 +08:00）。
- best.pth：[best.pth](</home/pai/Huawei/experiments/P4Transformer_T3_300/20261028_063154/checkpoint/best.pth>)（更新于 2026-10-28 21:36:27 +08:00）。

### P4Transformer bbox 扰动对比

从头训练 300 epoch，比较 bbox 扰动增强与原始 300 epoch 基线。

- 状态：已完成；阶段：未记录；已完成 300/300 epoch。
- 配置：[train.yaml](</home/pai/Huawei/codex/p4_bbox_jitter_compare/train.yaml>)；状态依据：[job.json](</home/pai/Huawei/codex/p4_bbox_jitter_compare/job.json>)。
- 最近完整 epoch：300（2026-10-29 04:33:06 +08:00）；训练 MPJPE：7.930 cm；验证 MPJPE：15.531 cm；验证 loss：0.15531237584904653。
- 已记录的最佳验证 MPJPE：15.515 cm（epoch 272）。
- 训练日志：[log.txt](</home/pai/Huawei/experiments/P4Transformer/20261028_161946/log/log.txt>)。
- 控制台：[train_console.log](</home/pai/Huawei/codex/p4_bbox_jitter_compare/train_console.log>)。
- 最近证据更新时间：2026-10-30 14:01:33 +08:00。
- 实验目录：[20261028_161946](</home/pai/Huawei/experiments/P4Transformer/20261028_161946>)。
- last.pth：[last.pth](</home/pai/Huawei/experiments/P4Transformer/20261028_161946/checkpoint/last.pth>)（更新于 2026-10-30 14:01:33 +08:00）。
- best.pth：[best.pth](</home/pai/Huawei/experiments/P4Transformer/20261028_161946/checkpoint/best.pth>)（更新于 2026-10-30 10:47:34 +08:00）。

### P4Transformer MOCAP-21，步长1微调50轮

从 MOCAP 第300轮 last 恢复，使用16帧窗口、步长1微调50轮（301–350）；lr=0.0003、无预热、余弦50轮、batch=64、BF16。保留原实验，续训结果另存。

- 状态：运行中；阶段：training；已完成 324/350 epoch。
- 配置：[config.yaml](</home/pai/Huawei/experiments/P4Transformer_mocap_300/queue_stride1_100_20261028_213911/config.yaml>)；状态依据：[status.json](</home/pai/Huawei/experiments/P4Transformer_mocap_300/queue_stride1_100_20261028_213911/status.json>)。
- PID：246250；用户：pai；进程开始：2026-10-29 10:24:38 +08:00。
- 最近完整 epoch：324（2026-10-30 14:03:49 +08:00）；训练 MPJPE：6.317 cm；验证 MPJPE：13.347 cm；验证 loss：0.13347413527598645。
- 已记录的最佳验证 MPJPE：13.208 cm（epoch 322）。
- 预计训练结束：2026-10-31 13:09:24 +08:00（最近完整 epoch 耗时中位数，仅供参考，不含后续评估）。
- 训练日志：[log.txt](</home/pai/Huawei/experiments/P4Transformer_mocap_300/queue_stride1_100_20261028_213911/experiment/log/log.txt>)。
- 控制台：[console.log](</home/pai/Huawei/experiments/P4Transformer_mocap_300/queue_stride1_100_20261028_213911/console.log>)。
- 最近证据更新时间：2026-10-30 14:30:01 +08:00。
- 实验目录：[experiment](</home/pai/Huawei/experiments/P4Transformer_mocap_300/queue_stride1_100_20261028_213911/experiment>)。
- last.pth：[last.pth](</home/pai/Huawei/experiments/P4Transformer_mocap_300/queue_stride1_100_20261028_213911/experiment/checkpoint/last.pth>)（更新于 2026-10-30 14:03:49 +08:00）。
- best.pth：[best.pth](</home/pai/Huawei/experiments/P4Transformer_mocap_300/queue_stride1_100_20261028_213911/experiment/checkpoint/best.pth>)（更新于 2026-10-30 12:44:31 +08:00）。

### P4Transformer xyz，步长1微调50轮

保留 xyz-only 输入和 COCO-17 输出，从第300轮 last 恢复，使用16帧窗口、步长1微调50轮（301–350）；lr=0.0003、无预热、余弦50轮、batch=64、BF16。

- 状态：排队中；阶段：waiting_predecessor；已完成 300/350 epoch。
- 配置：[config.yaml](</home/pai/Huawei/experiments/P4Transformer_xyz_300/queue_stride1_100_20261028_214542/config.yaml>)；状态依据：[status.json](</home/pai/Huawei/experiments/P4Transformer_xyz_300/queue_stride1_100_20261028_214542/status.json>)。
- PID：999006；用户：pai；进程开始：2026-10-30 09:52:09 +08:00。
- 控制台：尚未生成。
- 等待日志：[queue.log](</home/pai/Huawei/experiments/P4Transformer_xyz_300/queue_stride1_100_20261028_214542/queue.log>)。
- 最近证据更新时间：2026-10-30 09:52:09 +08:00。
- 实验目录：[experiment](</home/pai/Huawei/experiments/P4Transformer_xyz_300/queue_stride1_100_20261028_214542/experiment>)。
- last.pth：[last.pth](</home/pai/Huawei/experiments/P4Transformer_xyz_300/queue_stride1_100_20261028_214542/experiment/checkpoint/last.pth>)（更新于 2026-10-27 04:00:09 +08:00）。

### P4Transformer xyzv，步长1微调50轮

保留 xyzv 输入和 COCO-17 输出，从当前第259轮 last 恢复，使用16帧窗口、步长1微调50轮（260–309）；lr=0.0003、无预热、余弦50轮、batch=64、BF16。

- 状态：排队中；阶段：waiting_predecessor；已完成 259/309 epoch。
- 配置：[config.yaml](</home/pai/Huawei/experiments/P4Transformer_xyzv_300/queue_stride1_100_20261028_214542/config.yaml>)；状态依据：[status.json](</home/pai/Huawei/experiments/P4Transformer_xyzv_300/queue_stride1_100_20261028_214542/status.json>)。
- PID：246249；用户：pai；进程开始：2026-10-29 10:24:38 +08:00。
- 控制台：尚未生成。
- 等待日志：[queue.log](</home/pai/Huawei/experiments/P4Transformer_xyzv_300/queue_stride1_100_20261028_214542/queue.log>)。
- 最近证据更新时间：2026-10-29 10:24:37 +08:00。
- 实验目录：[experiment](</home/pai/Huawei/experiments/P4Transformer_xyzv_300/queue_stride1_100_20261028_214542/experiment>)。
- last.pth：[last.pth](</home/pai/Huawei/experiments/P4Transformer_xyzv_300/queue_stride1_100_20261028_214542/experiment/checkpoint/last.pth>)（更新于 2026-10-27 16:29:58 +08:00）。

### P4Transformer T=1 时间窗口消融

提供单帧消融结果，与 T=3 和原始基线比较。

- 状态：已完成；阶段：complete；已完成 300/300 epoch。
- 配置：[t1_effective.yaml](</home/pai/Huawei/@codex/p4_window_ablation_20261026/t1_effective.yaml>)；状态依据：[t1_job.json](</home/pai/Huawei/@codex/p4_window_ablation_20261026/t1_job.json>)。
- 最近完整 epoch：300（2026-10-28 06:31:50 +08:00）；训练 MPJPE：6.191 cm；验证 MPJPE：15.136 cm；验证 loss：0.15135524486324517。
- 已记录的最佳验证 MPJPE：15.048 cm（epoch 225）。
- 训练日志：[log.txt](</home/pai/Huawei/experiments/P4Transformer_T1_300/20261026_210459/log/log.txt>)。
- 控制台：[t1.log](</home/pai/Huawei/@codex/p4_window_ablation_20261026/t1.log>)。
- 最近证据更新时间：2026-10-28 06:31:51 +08:00。
- 实验目录：[20261026_210459](</home/pai/Huawei/experiments/P4Transformer_T1_300/20261026_210459>)。
- last.pth：[last.pth](</home/pai/Huawei/experiments/P4Transformer_T1_300/20261026_210459/checkpoint/last.pth>)（更新于 2026-10-28 06:31:50 +08:00）。
- best.pth：[best.pth](</home/pai/Huawei/experiments/P4Transformer_T1_300/20261026_210459/checkpoint/best.pth>)（更新于 2026-10-27 22:21:52 +08:00）。

### ConvNeXtV2 xyz-only，50 epoch

已启动的多模型 xyz-only 比较中的 ConvNeXtV2 实验；保留失败记录，避免误认为仍在排队。

- 状态：已失败；阶段：training；已完成 0/50 epoch。
- 配置：[ConvNeXtV2.yaml](</home/pai/Huawei/codex/model_compare_xyz50/20261027_163513_488571/ConvNeXtV2.yaml>)；状态依据：[ConvNeXtV2.status.json](</home/pai/Huawei/codex/model_compare_xyz50/20261027_163513_488571/ConvNeXtV2.status.json>)。
- 训练日志：[log.txt](</home/pai/Huawei/experiments/modelcompare/ConvNeXtV2_xyz_50/20261028_041911/log/log.txt>)。
- 控制台：[ConvNeXtV2.log](</home/pai/Huawei/codex/model_compare_xyz50/20261027_163513_488571/ConvNeXtV2.log>)。
- 最近证据更新时间：2026-10-28 04:19:30 +08:00。
- 实验目录：[20261028_041911](</home/pai/Huawei/experiments/modelcompare/ConvNeXtV2_xyz_50/20261028_041911>)。

### P4Transformer xyz 多模型对比，微调 51–100 epoch

恢复第 50 轮模型、AdamW 状态与指标历史，额外训练 50 轮。训练与验证均为 T=16、步长 1，累积 3 帧，batch=64；学习率从 0.003 开始、50 轮直接余弦衰减，无 warmup。日志追加到原实验，本阶段独立统计 best，旧检查点已备份。已完成7轮微调，从第57轮完整检查点补剩余43轮；历史前7轮保留，后续使用50轮余弦对应位置。

- 状态：排队中；阶段：waiting_priority_bbox；已完成 57/100 epoch。
- 配置：[P4Transformer.yaml](</home/pai/Huawei/experiments/modelcompare/queues/resume_stride1_20261028_2125/P4Transformer.yaml>)；状态依据：[P4Transformer.status.json](</home/pai/Huawei/experiments/modelcompare/queues/resume_stride1_20261028_2125/P4Transformer.status.json>)。
- 队列等待进程 PID：1007657；用户：pai；进程开始：2026-10-30 10:08:31 +08:00。
- 所属队列：[queue.status.json](</home/pai/Huawei/experiments/modelcompare/queues/resume_stride1_20261028_2125/queue.status.json>)；启动条件：第1/5项：等待当前bbox微调及最高优先级ResNet3D第25–48轮结束、旧暂停控制进程释放锁，再通过GPU0三次空闲检查。
- 最近完整 epoch：57（2026-10-29 10:14:03 +08:00）；训练 MPJPE：9.124 cm；验证 MPJPE：15.701 cm；验证 loss：0.15700530126514897。
- 已记录的最佳验证 MPJPE：15.693 cm（epoch 55）。
- 训练日志：[log.txt](</home/pai/Huawei/experiments/modelcompare/P4Transformer_xyz_50/20261027_163517/log/log.txt>)。
- 控制台：[console.log](</home/pai/Huawei/experiments/modelcompare/P4Transformer_xyz_50/20261027_163517/log/console.log>)。
- 等待日志：[queue.log](</home/pai/Huawei/experiments/modelcompare/queues/resume_stride1_20261028_2125/queue.log>)。
- 最近证据更新时间：2026-10-30 14:29:45 +08:00。
- 实验目录：[20261027_163517](</home/pai/Huawei/experiments/modelcompare/P4Transformer_xyz_50/20261027_163517>)。
- last.pth：[last.pth](</home/pai/Huawei/experiments/modelcompare/P4Transformer_xyz_50/20261027_163517/checkpoint/last.pth>)（更新于 2026-10-29 10:14:03 +08:00）。
- best.pth：[best.pth](</home/pai/Huawei/experiments/modelcompare/P4Transformer_xyz_50/20261027_163517/checkpoint/best.pth>)（更新于 2026-10-29 08:36:31 +08:00）。

### PointNet xyz 多模型对比，微调 51–100 epoch

恢复第 50 轮模型、AdamW 状态与指标历史，额外训练 50 轮。训练与验证均为 T=16、步长 1，累积 3 帧，batch=64；学习率从 0.003 开始、50 轮直接余弦衰减，无 warmup。日志追加到原实验，本阶段独立统计 best，旧检查点已备份。

- 状态：排队中；阶段：waiting_priority_bbox；已完成 50/100 epoch。
- 配置：[PointNet.yaml](</home/pai/Huawei/experiments/modelcompare/queues/resume_stride1_20261028_2125/PointNet.yaml>)；状态依据：[PointNet.status.json](</home/pai/Huawei/experiments/modelcompare/queues/resume_stride1_20261028_2125/PointNet.status.json>)。
- 队列等待进程 PID：1007657；用户：pai；进程开始：2026-10-30 10:08:31 +08:00。
- 所属队列：[queue.status.json](</home/pai/Huawei/experiments/modelcompare/queues/resume_stride1_20261028_2125/queue.status.json>)；启动条件：同一 GPU0 串行队列第 2/5 项；等待 P4Transformer 本阶段训练完成。
- 训练日志：[log.txt](</home/pai/Huawei/experiments/modelcompare/PointNet_xyz_50/20261027_190445/log/log.txt>)。
- 控制台：[console.log](</home/pai/Huawei/experiments/modelcompare/PointNet_xyz_50/20261027_190445/log/console.log>)。
- 等待日志：[queue.log](</home/pai/Huawei/experiments/modelcompare/queues/resume_stride1_20261028_2125/queue.log>)。
- 最近证据更新时间：2026-10-30 14:29:45 +08:00。
- 实验目录：[20261027_190445](</home/pai/Huawei/experiments/modelcompare/PointNet_xyz_50/20261027_190445>)。
- last.pth：[last.pth](</home/pai/Huawei/experiments/modelcompare/PointNet_xyz_50/20261027_190445/checkpoint/last.pth>)（更新于 2026-10-27 20:37:38 +08:00）。
- best.pth：[best.pth](</home/pai/Huawei/experiments/modelcompare/PointNet_xyz_50/20261027_190445/checkpoint/best.pth>)（更新于 2026-10-27 20:24:50 +08:00）。

### SetTransformer xyz 多模型对比，微调 51–100 epoch

恢复第 50 轮模型、AdamW 状态与指标历史，额外训练 50 轮。训练与验证均为 T=16、步长 1，累积 3 帧，batch=64；学习率从 0.003 开始、50 轮直接余弦衰减，无 warmup。日志追加到原实验，本阶段独立统计 best，旧检查点已备份。

- 状态：排队中；阶段：waiting_priority_bbox；已完成 50/100 epoch。
- 配置：[SetTransformer.yaml](</home/pai/Huawei/experiments/modelcompare/queues/resume_stride1_20261028_2125/SetTransformer.yaml>)；状态依据：[SetTransformer.status.json](</home/pai/Huawei/experiments/modelcompare/queues/resume_stride1_20261028_2125/SetTransformer.status.json>)。
- 队列等待进程 PID：1007657；用户：pai；进程开始：2026-10-30 10:08:31 +08:00。
- 所属队列：[queue.status.json](</home/pai/Huawei/experiments/modelcompare/queues/resume_stride1_20261028_2125/queue.status.json>)；启动条件：同一 GPU0 串行队列第 3/5 项；等待 PointNet 本阶段训练完成。
- 训练日志：[log.txt](</home/pai/Huawei/experiments/modelcompare/SetTransformer_xyz_50/20261028_004422/log/log.txt>)。
- 控制台：[console.log](</home/pai/Huawei/experiments/modelcompare/SetTransformer_xyz_50/20261028_004422/log/console.log>)。
- 等待日志：[queue.log](</home/pai/Huawei/experiments/modelcompare/queues/resume_stride1_20261028_2125/queue.log>)。
- 最近证据更新时间：2026-10-30 14:29:45 +08:00。
- 实验目录：[20261028_004422](</home/pai/Huawei/experiments/modelcompare/SetTransformer_xyz_50/20261028_004422>)。
- last.pth：[last.pth](</home/pai/Huawei/experiments/modelcompare/SetTransformer_xyz_50/20261028_004422/checkpoint/last.pth>)（更新于 2026-10-28 02:39:21 +08:00）。
- best.pth：[best.pth](</home/pai/Huawei/experiments/modelcompare/SetTransformer_xyz_50/20261028_004422/checkpoint/best.pth>)（更新于 2026-10-28 02:39:21 +08:00）。

### VoxelNeXt xyz 多模型对比，微调 51–100 epoch

恢复第 50 轮模型、AdamW 状态与指标历史，额外训练 50 轮。训练与验证均为 T=16、步长 1，累积 3 帧，batch=64；学习率从 0.003 开始、50 轮直接余弦衰减，无 warmup。日志追加到原实验，本阶段独立统计 best，旧检查点已备份。

- 状态：排队中；阶段：waiting_priority_bbox；已完成 50/100 epoch。
- 配置：[VoxelNeXt.yaml](</home/pai/Huawei/experiments/modelcompare/queues/resume_stride1_20261028_2125/VoxelNeXt.yaml>)；状态依据：[VoxelNeXt.status.json](</home/pai/Huawei/experiments/modelcompare/queues/resume_stride1_20261028_2125/VoxelNeXt.status.json>)。
- 队列等待进程 PID：1007657；用户：pai；进程开始：2026-10-30 10:08:31 +08:00。
- 所属队列：[queue.status.json](</home/pai/Huawei/experiments/modelcompare/queues/resume_stride1_20261028_2125/queue.status.json>)；启动条件：同一 GPU0 串行队列第 4/5 项；等待 SetTransformer 本阶段训练完成。
- 训练日志：[log.txt](</home/pai/Huawei/experiments/modelcompare/VoxelNeXt_xyz_50/20261028_023946/log/log.txt>)。
- 控制台：[console.log](</home/pai/Huawei/experiments/modelcompare/VoxelNeXt_xyz_50/20261028_023946/log/console.log>)。
- 等待日志：[queue.log](</home/pai/Huawei/experiments/modelcompare/queues/resume_stride1_20261028_2125/queue.log>)。
- 最近证据更新时间：2026-10-30 14:29:45 +08:00。
- 实验目录：[20261028_023946](</home/pai/Huawei/experiments/modelcompare/VoxelNeXt_xyz_50/20261028_023946>)。
- last.pth：[last.pth](</home/pai/Huawei/experiments/modelcompare/VoxelNeXt_xyz_50/20261028_023946/checkpoint/last.pth>)（更新于 2026-10-28 04:19:04 +08:00）。
- best.pth：[best.pth](</home/pai/Huawei/experiments/modelcompare/VoxelNeXt_xyz_50/20261028_023946/checkpoint/best.pth>)（更新于 2026-10-28 04:19:04 +08:00）。

### ConvNeXtV2 xyz 多模型对比，从头训练 100 epoch

随机初始化训练 100 轮，无初始化检查点；ConvNeXtV2 使用每块 32 帧与梯度检查点降低显存占用。训练与验证均为 T=16、步长 1，累积 3 帧，batch=64；学习率从 0.003 开始、100 轮直接余弦衰减，无 warmup。日志追加到原实验，本阶段独立统计 best，旧检查点已备份。

- 状态：排队中；阶段：priority_resnet3d；已完成 0/100 epoch。
- 配置：[ConvNeXtV2.yaml](</home/pai/Huawei/experiments/modelcompare/queues/resume_stride1_20261028_2125/ConvNeXtV2.yaml>)；状态依据：[ConvNeXtV2.status.json](</home/pai/Huawei/experiments/modelcompare/queues/resume_stride1_20261028_2125/ConvNeXtV2.status.json>)。
- 队列等待进程 PID：1007657；用户：pai；进程开始：2026-10-30 10:08:31 +08:00。
- 所属队列：[queue.status.json](</home/pai/Huawei/experiments/modelcompare/queues/resume_stride1_20261028_2125/queue.status.json>)；启动条件：同一 GPU0 串行队列第 5/5 项；等待 VoxelNeXt 本阶段训练完成。
- 训练日志：[log.txt](</home/pai/Huawei/experiments/modelcompare/ConvNeXtV2_xyz_50/20261028_041911/log/log.txt>)。
- 控制台：[console.log](</home/pai/Huawei/experiments/modelcompare/ConvNeXtV2_xyz_50/20261028_041911/log/console.log>)。
- 等待日志：[queue.log](</home/pai/Huawei/experiments/modelcompare/queues/resume_stride1_20261028_2125/queue.log>)。
- 最近证据更新时间：2026-10-30 14:29:45 +08:00。
- 实验目录：[20261028_041911](</home/pai/Huawei/experiments/modelcompare/ConvNeXtV2_xyz_50/20261028_041911>)。

