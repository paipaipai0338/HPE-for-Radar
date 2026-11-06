# 训练监控

最后刷新：2026-11-06 14:30:01 +08:00；刷新间隔：10 分钟；采集状态：成功。任务 / 采集提示：0 项，见下文。

下次计划刷新：2026-11-06 14:40:00 +08:00。超过 20 分钟未更新请检查 [health.json](</home/pai/Huawei/supervise/health.json>) 和 [monitor.log](</home/pai/Huawei/supervise/monitor.log>)。

范围：tasks.json 登记的项目任务；GPU 表列出全机计算占用。MPJPE 统一为 cm；不同骨架与数据划分的指标不宜直接横比。

## 当前任务

| 任务 | 状态 | GPU | 已完成 epoch | 最近验证 MPJPE | 预计结束时间 |
| --- | --- | --- | --- | --- | --- |
| ResNet3D 全量续训49–58轮 | 运行中 | 1 | 52/58 | 14.411 cm | 2026-11-07 13:59:27 +08:00 |
| FFTGen 数据处理后重跑：原best stride1微调50轮 | 运行中 | 0 | 44/71 | 13.176 cm | 2026-11-07 06:46:45 +08:00 |

预计结束时间为北京时间的训练结束估计，不含后续评估；最多取最近 11 条完整 epoch 日志，有多少用多少。只有一条时按进程启动至该轮完成的耗时估算（含初始化）；无有效耗时数据或正在评估时显示“暂无法估算”。

## 排队与后续任务

| 任务 | 状态 | GPU | 启动条件 / 内容 |
| --- | --- | --- | --- |
| 暂无已登记排队任务 | — | — | — |

顺序来自现有任务和启动条件；本监控不负责启动、重试或调整训练。

## GPU 占用

| GPU | 型号 | 利用率 | 显存 MiB（已用 / 总计） |
| --- | --- | --- | --- |
| 0 | NVIDIA GeForce RTX 5090 | 43% | 7489 / 32607 |
| 1 | NVIDIA GeForce RTX 5090 | 95% | 12999 / 32607 |

| GPU | PID | 用户 | 归属 | 命令 |
| --- | --- | --- | --- | --- |
| 0 | 78754 | pai | 已登记任务 | /home/pai/miniconda3/envs/pytorch/bin/python -B -u /home/pai/Huawei/experiments/FFTGen/P4Transformer/20261105_203931/launch.py worker |
| 1 | 45380 | pai | 已登记任务 | /home/pai/miniconda3/envs/pytorch/bin/python -B -u /home/pai/Huawei/experiments/ResNet3D/20261030_100710/full_train_20261105/launch.py worker |
| 1 | 138980 | pai | 未登记占用（不推断实验配置） | /home/pai/miniconda3/envs/pytorch/bin/python -B -u -i |

## 异常与待处理

- 本次未发现已登记任务异常。

## 已完成、失败或待核实的登记任务

| 任务 | 状态 | GPU | 已完成 epoch | 最近验证 MPJPE |
| --- | --- | --- | --- | --- |
| ResNet3D 最佳版续训25–48轮，GPU0排队首位 | 已完成 | 0 | 48/48 | 14.486 cm |
| P4Transformer bbox扰动，GPU0最高优先级微调50轮 | 已完成 | 0 | 350/350 | 14.836 cm |
| P4Transformer MOCAP-21，300 epoch | 已完成 | 0 | 300/300 | 14.209 cm |
| P4Transformer T=3 时间窗口消融 | 已完成 | 1 | 300/300 | 14.618 cm |
| P4Transformer bbox 扰动对比 | 已完成 | 0 | 300/300 | 15.531 cm |
| P4Transformer MOCAP-21，步长1微调50轮 | 已完成 | 1 | 350/350 | 13.113 cm |
| P4Transformer xyz，步长1微调50轮 | 已完成 | 1 | 350/350 | 14.730 cm |
| P4Transformer T=1 时间窗口消融 | 已完成 | 1 | 300/300 | 15.136 cm |
| ConvNeXtV2 xyz-only，50 epoch | 已由后续任务接替 | 0 | 0/50 | 未记录 |
| P4Transformer xyz 多模型对比，微调 51–100 epoch | 已完成 | 0 | 100/100 | 14.225 cm |
| PointNet xyz 多模型对比，微调 51–100 epoch | 已中断 | 0 | 70/100 | 786.257 cm |
| SetTransformer xyz 多模型对比，微调 51–100 epoch | 已完成 | 0 | 100/100 | 16.056 cm |
| VoxelNeXt xyz，单体素修复后微调51–100轮 | 已中断 | 0 | 50/100 | 未记录 |
| P4Transformer xyzv，缓存修复后续训295–309轮 | 已完成 | 1 | 309/309 | 14.252 cm |
| ConvNeXtV2 xyz 多模型对比，从头训练 100 epoch | 已中断 | 0 | 4/100 | 140.721 cm |
| FFTGen P4Transformer xyz，MOCAP-21，500点，100轮 | 已完成 | 1 | 100/100 | 24.917 cm |
| FFTGen best第77轮续训30轮（78–107），GPU1 | 已完成 | 1 | 107/107 | 17.157 cm |
| FFTGen 原版best第21轮，stride1微调50轮 | 已完成 | 0 | 71/71 | 13.435 cm |

## 任务详情与证据

### ResNet3D 全量续训49–58轮

从第48轮last恢复模型、AdamW及指标历史；取消8000批有放回抽样，每轮完整打乱遍历911543个训练窗口（56972批，保留尾批），验证190015个窗口。沿用原504/221组清单、T=1、stride=1、batch16、BF16、COCO17、DXYZ+KXYZ及原增强/骨长约束。学习率重设1e-5，10轮余弦、无预热；保留原第38轮best及144.69mm门槛。独立目录保存续训结果。

- 状态：运行中；阶段：training；已完成 52/58 epoch。
- 配置：[config.yaml](</home/pai/Huawei/experiments/ResNet3D/20261030_100710/full_train_20261105/config.yaml>)；状态依据：[status.json](</home/pai/Huawei/experiments/ResNet3D/20261030_100710/full_train_20261105/status.json>)。
- PID：45380；用户：pai；进程开始：2026-11-05 18:55:39 +08:00。
- 最近完整 epoch：52（2026-11-06 12:00:39 +08:00）；训练 MPJPE：9.340 cm；验证 MPJPE：14.411 cm；验证 loss：0.14410690730028966。
- 预计训练结束：2026-11-07 13:59:27 +08:00（最近完整 epoch 耗时中位数，仅供参考，不含后续评估）。
- 训练日志：[log.txt](</home/pai/Huawei/experiments/ResNet3D/20261030_100710/full_train_20261105/log/log.txt>)。
- 控制台：[console.log](</home/pai/Huawei/experiments/ResNet3D/20261030_100710/full_train_20261105/console.log>)。
- 最近证据更新时间：2026-11-06 14:30:01 +08:00。
- 实验目录：[full_train_20261105](</home/pai/Huawei/experiments/ResNet3D/20261030_100710/full_train_20261105>)。
- last.pth：[last.pth](</home/pai/Huawei/experiments/ResNet3D/20261030_100710/full_train_20261105/checkpoint/last.pth>)（更新于 2026-11-06 12:00:39 +08:00）。
- best.pth：[best.pth](</home/pai/Huawei/experiments/ResNet3D/20261030_100710/full_train_20261105/checkpoint/best.pth>)（更新于 2026-11-06 12:00:39 +08:00）。

### ResNet3D 最佳版续训25–48轮，GPU0排队首位

从20261023_065632第24轮last恢复权重、AdamW及指标历史，在当前数据清单（请求训练504组/验证221组，含俯卧）继续到48轮；COCO-17、DXYZ+KXYZ、GT ROI、T=1、步长1、batch16、BF16、每轮8000批；平移±10cm、镜像50%、功率增强、骨长约束0.2；lr=0.00004、余弦24轮、无预热。新验证清单独立统计best，历史实验保留。

- 状态：已完成；阶段：complete；已完成 48/48 epoch。
- 配置：[config.yaml](</home/pai/Huawei/experiments/ResNet3D/20261030_100710/config/config.yaml>)；状态依据：[status.json](</home/pai/Huawei/experiments/ResNet3D/20261030_100710/continuation/status.json>)。
- 最近完整 epoch：48（2026-10-31 19:48:03 +08:00）；训练 MPJPE：9.808 cm；验证 MPJPE：14.486 cm；验证 loss：0.1448594970605182。
- 训练日志：[log.txt](</home/pai/Huawei/experiments/ResNet3D/20261030_100710/log/log.txt>)。
- 控制台：[console.log](</home/pai/Huawei/experiments/ResNet3D/20261030_100710/continuation/console.log>)。
- 最近证据更新时间：2026-10-31 19:48:05 +08:00。
- 实验目录：[20261030_100710](</home/pai/Huawei/experiments/ResNet3D/20261030_100710>)。
- last.pth：[last.pth](</home/pai/Huawei/experiments/ResNet3D/20261030_100710/checkpoint/last.pth>)（更新于 2026-10-31 19:48:03 +08:00）。
- best.pth：[best.pth](</home/pai/Huawei/experiments/ResNet3D/20261030_100710/checkpoint/best.pth>)（更新于 2026-10-31 10:19:57 +08:00）。

### P4Transformer bbox扰动，GPU0最高优先级微调50轮

从20261028_161946第300轮last恢复，T=16、训练/验证步长1，微调301–350轮；lr=0.0003、无预热、余弦50轮、batch=64、BF16，保留每帧bbox扰动和跟踪框增强。日志、权重和配置续写原实验目录；300轮结果备份于backup_before_stride1_50。原GPU0多模型队列整体内存暂停，本任务退出后自动原地恢复。

- 状态：已完成；阶段：complete；已完成 350/350 epoch。
- 配置：[resume_stride1_350.yaml](</home/pai/Huawei/experiments/P4Transformer/20261028_161946/config/resume_stride1_350.yaml>)；状态依据：[status.json](</home/pai/Huawei/experiments/P4Transformer/20261028_161946/finetune_stride1_50/status.json>)。
- 最近完整 epoch：350（2026-10-30 20:17:13 +08:00）；训练 MPJPE：6.600 cm；验证 MPJPE：14.836 cm；验证 loss：0.14836068025475488。
- 已记录的最佳验证 MPJPE：14.813 cm（epoch 335）。
- 训练日志：[log.txt](</home/pai/Huawei/experiments/P4Transformer/20261028_161946/log/log.txt>)。
- 控制台：[finetune_console.log](</home/pai/Huawei/experiments/P4Transformer/20261028_161946/log/finetune_console.log>)。
- 最近证据更新时间：2026-10-30 20:17:16 +08:00。
- 实验目录：[20261028_161946](</home/pai/Huawei/experiments/P4Transformer/20261028_161946>)。
- last.pth：[last.pth](</home/pai/Huawei/experiments/P4Transformer/20261028_161946/checkpoint/last.pth>)（更新于 2026-10-30 20:17:13 +08:00）。
- best.pth：[best.pth](</home/pai/Huawei/experiments/P4Transformer/20261028_161946/checkpoint/best.pth>)（更新于 2026-10-30 10:47:34 +08:00）。

### P4Transformer MOCAP-21，300 epoch

保持原训练条件，将 COCO-17 改为 MOCAP-21，训练后比较骨架插值精度。

- 状态：已完成；阶段：complete；已完成 300/300 epoch。
- 配置：[config.yaml](</home/pai/Huawei/experiments/P4Transformer_mocap_300/20261028_042133/initial_300/config.yaml>)；状态依据：[status.json](</home/pai/Huawei/experiments/P4Transformer_mocap_300/20261028_042133/initial_300/status.json>)。
- 最近完整 epoch：300（2026-10-28 16:17:57 +08:00）；训练 MPJPE：6.959 cm；验证 MPJPE：14.209 cm；验证 loss：0.14209393539928178。
- 已记录的最佳验证 MPJPE：14.203 cm（epoch 292）。
- 训练日志：[log.txt](</home/pai/Huawei/experiments/P4Transformer_mocap_300/20261028_042133/initial_300/20261028_042133/log/log.txt>)。
- 控制台：[console.log](</home/pai/Huawei/experiments/P4Transformer_mocap_300/20261028_042133/initial_300/console.log>)。
- 最近证据更新时间：2026-11-03 10:00:33 +08:00。
- 实验目录：[20261028_042133](</home/pai/Huawei/experiments/P4Transformer_mocap_300/20261028_042133/initial_300/20261028_042133>)。
- last.pth：[last.pth](</home/pai/Huawei/experiments/P4Transformer_mocap_300/20261028_042133/initial_300/20261028_042133/checkpoint/last.pth>)（更新于 2026-10-28 16:17:57 +08:00）。
- best.pth：[best.pth](</home/pai/Huawei/experiments/P4Transformer_mocap_300/20261028_042133/initial_300/20261028_042133/checkpoint/best.pth>)（更新于 2026-10-28 15:59:21 +08:00）。

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
- 最近证据更新时间：2026-10-30 20:17:13 +08:00。
- 实验目录：[20261028_161946](</home/pai/Huawei/experiments/P4Transformer/20261028_161946>)。
- last.pth：[last.pth](</home/pai/Huawei/experiments/P4Transformer/20261028_161946/checkpoint/last.pth>)（更新于 2026-10-30 20:17:13 +08:00）。
- best.pth：[best.pth](</home/pai/Huawei/experiments/P4Transformer/20261028_161946/checkpoint/best.pth>)（更新于 2026-10-30 10:47:34 +08:00）。

### P4Transformer MOCAP-21，步长1微调50轮

从 MOCAP 第300轮 last 恢复，使用16帧窗口、步长1微调50轮（301–350）；lr=0.0003、无预热、余弦50轮、batch=64、BF16。保留原实验，续训结果另存。

- 状态：已完成；阶段：complete；已完成 350/350 epoch。
- 配置：[config.yaml](</home/pai/Huawei/experiments/P4Transformer_mocap_300/20261028_042133/finetune_stride1_50/config.yaml>)；状态依据：[status.json](</home/pai/Huawei/experiments/P4Transformer_mocap_300/20261028_042133/finetune_stride1_50/status.json>)。
- 最近完整 epoch：350（2026-10-31 08:30:02 +08:00）；训练 MPJPE：5.704 cm；验证 MPJPE：13.113 cm；验证 loss：0.13112522576043695。
- 已记录的最佳验证 MPJPE：13.101 cm（epoch 346）。
- 训练日志：[log.txt](</home/pai/Huawei/experiments/P4Transformer_mocap_300/20261028_042133/log/log.txt>)。
- 控制台：[console.log](</home/pai/Huawei/experiments/P4Transformer_mocap_300/20261028_042133/finetune_stride1_50/console.log>)。
- 最近证据更新时间：2026-10-31 08:30:09 +08:00。
- 实验目录：[20261028_042133](</home/pai/Huawei/experiments/P4Transformer_mocap_300/20261028_042133>)。
- last.pth：[last.pth](</home/pai/Huawei/experiments/P4Transformer_mocap_300/20261028_042133/checkpoint/last.pth>)（更新于 2026-10-31 08:30:02 +08:00）。
- best.pth：[best.pth](</home/pai/Huawei/experiments/P4Transformer_mocap_300/20261028_042133/checkpoint/best.pth>)（更新于 2026-10-31 05:29:24 +08:00）。

### P4Transformer xyz，步长1微调50轮

保留 xyz-only 输入和 COCO-17 输出，从第300轮 last 恢复，使用16帧窗口、步长1微调50轮（301–350）；lr=0.0003、无预热、余弦50轮、batch=64、BF16。

- 状态：已完成；阶段：complete；已完成 350/350 epoch。
- 配置：[config.yaml](</home/pai/Huawei/experiments/P4Transformer_xyz_300/queue_stride1_100_20261028_214542/config.yaml>)；状态依据：[status.json](</home/pai/Huawei/experiments/P4Transformer_xyz_300/queue_stride1_100_20261028_214542/status.json>)。
- 最近完整 epoch：350（2026-11-01 17:03:26 +08:00）；训练 MPJPE：5.997 cm；验证 MPJPE：14.730 cm；验证 loss：0.14730390071198196。
- 已记录的最佳验证 MPJPE：14.621 cm（epoch 331）。
- 训练日志：[log.txt](</home/pai/Huawei/experiments/P4Transformer_xyz_300/queue_stride1_100_20261028_214542/experiment/log/log.txt>)。
- 控制台：[console.log](</home/pai/Huawei/experiments/P4Transformer_xyz_300/queue_stride1_100_20261028_214542/console.log>)。
- 最近证据更新时间：2026-11-03 10:00:33 +08:00。
- 实验目录：[experiment](</home/pai/Huawei/experiments/P4Transformer_xyz_300/queue_stride1_100_20261028_214542/experiment>)。
- last.pth：[last.pth](</home/pai/Huawei/experiments/P4Transformer_xyz_300/queue_stride1_100_20261028_214542/experiment/checkpoint/last.pth>)（更新于 2026-11-01 17:03:26 +08:00）。
- best.pth：[best.pth](</home/pai/Huawei/experiments/P4Transformer_xyz_300/queue_stride1_100_20261028_214542/experiment/checkpoint/best.pth>)（更新于 2026-11-01 05:18:15 +08:00）。

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

已启动的多模型 xyz-only 比较中的 ConvNeXtV2 实验；保留失败记录，避免误认为仍在排队。 此旧OOM任务已由GPU0的ConvNeXtV2 100轮分块/梯度检查点训练接替，不重复排队；原失败状态保留于 codex/model_compare_xyz50/20261027_163513_488571/ConvNeXtV2.status.json。

- 状态：已由后续任务接替；阶段：replaced_by_active_retry；已完成 0/50 epoch。
- 配置：[ConvNeXtV2.yaml](</home/pai/Huawei/codex/model_compare_xyz50/20261027_163513_488571/ConvNeXtV2.yaml>)；状态依据：[convnext_xyz50.resolved.json](</home/pai/Huawei/supervise/convnext_xyz50.resolved.json>)。
- 训练日志：[log.txt](</home/pai/Huawei/experiments/modelcompare/ConvNeXtV2_xyz_50/20261028_041911/log/log.txt>)。
- 控制台：[ConvNeXtV2.log](</home/pai/Huawei/codex/model_compare_xyz50/20261027_163513_488571/ConvNeXtV2.log>)。
- 最近证据更新时间：2026-11-04 15:50:04 +08:00。
- 实验目录：[20261028_041911](</home/pai/Huawei/experiments/modelcompare/ConvNeXtV2_xyz_50/20261028_041911>)。
- last.pth：[last.pth](</home/pai/Huawei/experiments/modelcompare/ConvNeXtV2_xyz_50/20261028_041911/checkpoint/last.pth>)（更新于 2026-11-04 14:49:39 +08:00）。
- best.pth：[best.pth](</home/pai/Huawei/experiments/modelcompare/ConvNeXtV2_xyz_50/20261028_041911/checkpoint/best.pth>)（更新于 2026-11-04 14:49:39 +08:00）。

### P4Transformer xyz 多模型对比，微调 51–100 epoch

恢复第 50 轮模型、AdamW 状态与指标历史，额外训练 50 轮。训练与验证均为 T=16、步长 1，累积 3 帧，batch=64；学习率从 0.003 开始、50 轮直接余弦衰减，无 warmup。日志追加到原实验，本阶段独立统计 best，旧检查点已备份。已完成7轮微调，从第57轮完整检查点补剩余43轮；历史前7轮保留，后续使用50轮余弦对应位置。

- 状态：已完成；阶段：complete；已完成 100/100 epoch。
- 配置：[P4Transformer.yaml](</home/pai/Huawei/experiments/modelcompare/queues/resume_stride1_20261028_2125/P4Transformer.yaml>)；状态依据：[P4Transformer.status.json](</home/pai/Huawei/experiments/modelcompare/queues/resume_stride1_20261028_2125/P4Transformer.status.json>)。
- 最近完整 epoch：100（2026-11-02 01:37:59 +08:00）；训练 MPJPE：5.661 cm；验证 MPJPE：14.225 cm；验证 loss：0.14224674663160075。
- 已记录的最佳验证 MPJPE：14.200 cm（epoch 99）。
- 训练日志：[log.txt](</home/pai/Huawei/experiments/modelcompare/P4Transformer_xyz_50/20261027_163517/log/log.txt>)。
- 控制台：[console.log](</home/pai/Huawei/experiments/modelcompare/P4Transformer_xyz_50/20261027_163517/log/console.log>)。
- 最近证据更新时间：2026-11-02 01:38:00 +08:00。
- 实验目录：[20261027_163517](</home/pai/Huawei/experiments/modelcompare/P4Transformer_xyz_50/20261027_163517>)。
- last.pth：[last.pth](</home/pai/Huawei/experiments/modelcompare/P4Transformer_xyz_50/20261027_163517/checkpoint/last.pth>)（更新于 2026-11-02 01:37:59 +08:00）。
- best.pth：[best.pth](</home/pai/Huawei/experiments/modelcompare/P4Transformer_xyz_50/20261027_163517/checkpoint/best.pth>)（更新于 2026-11-02 00:47:23 +08:00）。

### PointNet xyz 多模型对比，微调 51–100 epoch

PointNet 高学习率阶段已中断并完整备份，回退至恶化前的第 50 轮检查点。恢复第 50 轮模型、AdamW 状态与指标历史，额外训练 50 轮至 epoch 100。训练与验证均为 T=16、步长 1，累积 3 帧，batch=64；学习率从 0.0003 开始，50 轮余弦衰减，无 warmup。本次调整后的阶段独立统计 best，监督指标从新阶段标记开始。异常阶段备份：/home/pai/Huawei/experiments/modelcompare/PointNet_xyz_50/20261027_190445/backup_before_lr0p0003_20261102_110222。 用户已主动中断本次训练，最后完整 epoch 为 70；不再续训。

- 状态：已中断；阶段：stopped_by_user；已完成 70/100 epoch。
- 配置：[PointNet.yaml](</home/pai/Huawei/experiments/modelcompare/queues/resume_stride1_20261028_2125/PointNet.yaml>)；状态依据：[PointNet.status.json](</home/pai/Huawei/experiments/modelcompare/queues/resume_stride1_20261028_2125/PointNet.status.json>)。
- 最近完整 epoch：70（2026-11-02 21:20:13 +08:00）；训练 MPJPE：19.006 cm；验证 MPJPE：786.257 cm；验证 loss：7.862565276555359。
- 已记录的最佳验证 MPJPE：115.716 cm（epoch 57）。
- 训练日志：[log.txt](</home/pai/Huawei/experiments/modelcompare/PointNet_xyz_50/20261027_190445/log/log.txt>)。
- 控制台：[console.log](</home/pai/Huawei/experiments/modelcompare/PointNet_xyz_50/20261027_190445/log/console.log>)。
- 最近证据更新时间：2026-11-02 21:35:28 +08:00。
- 实验目录：[20261027_190445](</home/pai/Huawei/experiments/modelcompare/PointNet_xyz_50/20261027_190445>)。
- last.pth：[last.pth](</home/pai/Huawei/experiments/modelcompare/PointNet_xyz_50/20261027_190445/checkpoint/last.pth>)（更新于 2026-11-02 21:20:13 +08:00）。
- best.pth：[best.pth](</home/pai/Huawei/experiments/modelcompare/PointNet_xyz_50/20261027_190445/checkpoint/best.pth>)（更新于 2026-11-02 14:33:30 +08:00）。

### SetTransformer xyz 多模型对比，微调 51–100 epoch

恢复第 50 轮模型、AdamW 状态与指标历史，额外训练 50 轮至 epoch 100。训练与验证均为 T=16、步长 1，累积 3 帧，batch=64；学习率从 0.0003 开始，50 轮余弦衰减，无 warmup。本次调整后的阶段独立统计 best，监督指标从新阶段标记开始。 逐组报告接入真实时序Guard：Raw/Guard均按完整窗口末帧计分，分别记录全部/椭圆内误差和人帧数；loss与best选择保持原始Raw口径。

- 状态：已完成；阶段：complete；已完成 100/100 epoch。
- 配置：[SetTransformer.yaml](</home/pai/Huawei/experiments/modelcompare/queues/resume_stride1_20261028_2125/SetTransformer.yaml>)；状态依据：[SetTransformer.status.json](</home/pai/Huawei/experiments/modelcompare/queues/resume_stride1_20261028_2125/SetTransformer.status.json>)。
- 最近完整 epoch：100（2026-11-04 03:13:02 +08:00）；训练 MPJPE：7.320 cm；验证 MPJPE：16.056 cm；验证 loss：0.16056270272231982。
- 已记录的最佳验证 MPJPE：16.036 cm（epoch 96）。
- 训练日志：[log.txt](</home/pai/Huawei/experiments/modelcompare/SetTransformer_xyz_50/20261028_004422/log/log.txt>)。
- 控制台：[console.log](</home/pai/Huawei/experiments/modelcompare/SetTransformer_xyz_50/20261028_004422/log/console.log>)。
- 最近证据更新时间：2026-11-05 18:55:39 +08:00。
- 实验目录：[20261028_004422](</home/pai/Huawei/experiments/modelcompare/SetTransformer_xyz_50/20261028_004422>)。
- last.pth：[last.pth](</home/pai/Huawei/experiments/modelcompare/SetTransformer_xyz_50/20261028_004422/checkpoint/last.pth>)（更新于 2026-11-04 03:13:02 +08:00）。
- best.pth：[best.pth](</home/pai/Huawei/experiments/modelcompare/SetTransformer_xyz_50/20261028_004422/checkpoint/best.pth>)（更新于 2026-11-04 00:55:08 +08:00）。

### VoxelNeXt xyz，单体素修复后微调51–100轮

恢复第 50 轮模型、AdamW 状态与指标历史，额外训练 50 轮至 epoch 100。训练与验证均为 T=16、步长 1，累积 3 帧，batch=64；学习率从 0.0003 开始，50 轮余弦衰减，无 warmup。本次调整后的阶段独立统计 best，监督指标从新阶段标记开始。 逐组报告接入真实时序Guard：Raw/Guard均按完整窗口末帧计分，分别记录全部/椭圆内误差和人帧数；loss与best选择保持原始Raw口径。 2026-11-04已修复单有效体素导致的BatchNorm错误：单体素使用运行统计并保留梯度，其余输入保持原行为；从第50轮完整检查点重跑51–100轮，lr=0.0003、余弦50轮。完整模型BF16前向/反向、检查点恢复及缓存校验通过，失败现场已备份。 用户于2026-11-04终止，GPU0改跑原FFTGen best步长1微调。

- 状态：已中断；阶段：stopped_by_user；已完成 50/100 epoch。
- 配置：[VoxelNeXt.yaml](</home/pai/Huawei/experiments/modelcompare/queues/retry_voxelnext_20261104/VoxelNeXt.yaml>)；状态依据：[VoxelNeXt.status.json](</home/pai/Huawei/experiments/modelcompare/queues/retry_voxelnext_20261104/VoxelNeXt.status.json>)。
- 训练日志：[log.txt](</home/pai/Huawei/experiments/modelcompare/VoxelNeXt_xyz_50/20261028_023946/log/log.txt>)。
- 控制台：[console.log](</home/pai/Huawei/experiments/modelcompare/VoxelNeXt_xyz_50/20261028_023946/log/console.log>)。
- 最近证据更新时间：2026-11-05 14:21:28 +08:00。
- 实验目录：[20261028_023946](</home/pai/Huawei/experiments/modelcompare/VoxelNeXt_xyz_50/20261028_023946>)。
- last.pth：[last.pth](</home/pai/Huawei/experiments/modelcompare/VoxelNeXt_xyz_50/20261028_023946/checkpoint/last.pth>)（更新于 2026-10-28 04:19:04 +08:00）。
- best.pth：[best.pth](</home/pai/Huawei/experiments/modelcompare/VoxelNeXt_xyz_50/20261028_023946/checkpoint/best.pth>)（更新于 2026-10-28 04:19:04 +08:00）。

### P4Transformer xyzv，缓存修复后续训295–309轮

将原GPU1的原始PC xyzv、COCO-17、200点微调安排至GPU0 / GPU1中先空闲的卡；从第294轮完整检查点继续295–309轮，恢复模型、AdamW、指标历史与best，保持原50轮余弦调度位置，续跑lr=0.00006183221215612905，不重新预热或重置学习率。原P4模型及训练实现保持，逐组Raw/Guard报告接入共享Guard；第295轮未保存部分重跑。 2026-11-04已修复失效缓存软链接，重试目录指向仍保留的原始共享步长1缓存；训练/验证清单、窗口数、样本读取及第294轮模型/AdamW/余弦恢复检查通过。失败控制目录保留，原实验另有备份。

- 状态：已完成；阶段：complete；已完成 309/309 epoch。
- 配置：[config.yaml](</home/pai/Huawei/experiments/P4Transformer_xyzv_300/queue_stride1_100_20261028_214542/retry_cache_20261104/config.yaml>)；状态依据：[status.json](</home/pai/Huawei/experiments/P4Transformer_xyzv_300/queue_stride1_100_20261028_214542/retry_cache_20261104/status.json>)。
- 最近完整 epoch：309（2026-11-05 14:21:06 +08:00）；训练 MPJPE：5.885 cm；验证 MPJPE：14.252 cm；验证 loss：0.142515260515317。
- 已记录的最佳验证 MPJPE：14.222 cm（epoch 306）。
- 训练日志：[log.txt](</home/pai/Huawei/experiments/P4Transformer_xyzv_300/queue_stride1_100_20261028_214542/experiment/log/log.txt>)。
- 控制台：[console.log](</home/pai/Huawei/experiments/P4Transformer_xyzv_300/queue_stride1_100_20261028_214542/retry_cache_20261104/console.log>)。
- 最近证据更新时间：2026-11-05 14:21:28 +08:00。
- 实验目录：[experiment](</home/pai/Huawei/experiments/P4Transformer_xyzv_300/queue_stride1_100_20261028_214542/experiment>)。
- last.pth：[last.pth](</home/pai/Huawei/experiments/P4Transformer_xyzv_300/queue_stride1_100_20261028_214542/experiment/checkpoint/last.pth>)（更新于 2026-11-05 14:21:06 +08:00）。
- best.pth：[best.pth](</home/pai/Huawei/experiments/P4Transformer_xyzv_300/queue_stride1_100_20261028_214542/experiment/checkpoint/best.pth>)（更新于 2026-11-02 12:25:53 +08:00）。

### ConvNeXtV2 xyz 多模型对比，从头训练 100 epoch

随机初始化训练 100 轮，无初始化检查点；ConvNeXtV2 使用每块 32 帧与梯度检查点降低显存占用。训练与验证均为 T=16、步长 1，累积 3 帧，batch=64；学习率从 0.003 开始、100 轮直接余弦衰减，无 warmup。日志追加到原实验，本阶段独立统计 best，旧检查点已备份。 逐组报告接入真实时序Guard：Raw/Guard均按完整窗口末帧计分，分别记录全部/椭圆内误差和人帧数；loss与best选择保持原始Raw口径。 用户于2026-11-04主动停止训练；最后完整检查点为第4轮，第5轮未保存进度放弃，不自动续训。

- 状态：已中断；阶段：stopped_by_user；已完成 4/100 epoch。
- 配置：[ConvNeXtV2.yaml](</home/pai/Huawei/experiments/modelcompare/queues/resume_stride1_20261028_2125/ConvNeXtV2.yaml>)；状态依据：[ConvNeXtV2.status.json](</home/pai/Huawei/experiments/modelcompare/queues/resume_stride1_20261028_2125/ConvNeXtV2.status.json>)。
- 最近完整 epoch：4（2026-11-04 14:49:39 +08:00）；训练 MPJPE：129.656 cm；验证 MPJPE：140.721 cm；验证 loss：1.4072142433467982。
- 已记录的最佳验证 MPJPE：140.721 cm（epoch 4）。
- 训练日志：[log.txt](</home/pai/Huawei/experiments/modelcompare/ConvNeXtV2_xyz_50/20261028_041911/log/log.txt>)。
- 控制台：[console.log](</home/pai/Huawei/experiments/modelcompare/ConvNeXtV2_xyz_50/20261028_041911/log/console.log>)。
- 最近证据更新时间：2026-11-04 15:50:44 +08:00。
- 实验目录：[20261028_041911](</home/pai/Huawei/experiments/modelcompare/ConvNeXtV2_xyz_50/20261028_041911>)。
- last.pth：[last.pth](</home/pai/Huawei/experiments/modelcompare/ConvNeXtV2_xyz_50/20261028_041911/checkpoint/last.pth>)（更新于 2026-11-04 14:49:39 +08:00）。
- best.pth：[best.pth](</home/pai/Huawei/experiments/modelcompare/ConvNeXtV2_xyz_50/20261028_041911/checkpoint/best.pth>)（更新于 2026-11-04 14:49:39 +08:00）。

### FFTGen P4Transformer xyz，MOCAP-21，500点，100轮

参考原始PC xyz-100配置，从头训练，改用FFTGen点云、xyz前三列、每人500点；MOCAP-21（GT按已有映射转换，模型直接输出21关节）、绝对高位雷达坐标、Positive_MPJPE；T=16、acc_frame=3、训练batch32、验证batch16、BF16、seed42。1–50轮训练/验证步长16，lr=0.001、warmup10轮后余弦；51–100轮训练/验证步长1，恢复模型/AdamW/指标历史，lr=0.003、无warmup、余弦50轮，本阶段重置best。归档训练/验证划分，FFTGen有效训练328组、验证173组；与历史PC有效窗口不同。 逐组报告接入真实时序Guard：Raw/Guard均按完整窗口末帧计分，分别记录全部/椭圆内误差和人帧数；loss与best选择保持原始Raw口径。 第1轮原batch64验证OOM后，按用户要求缩小batch并从seed42重跑；失败实验保留。 再次OOM后保留训练batch32/验证16；注意力换用PyTorch SDPA减少二次方矩阵显存，从第1轮检查点恢复模型、AdamW、调度和指标历史。原队列目录缺失，控制目录恢复到实验内部 recovery_20261102_oom。

- 状态：已完成；阶段：complete；已完成 100/100 epoch。
- 配置：[phase1.yaml](</home/pai/Huawei/experiments/FFTGen/P4Transformer_xyz_mocap_100/P4Transformer/20261102_140828/recovery_20261102_oom/phase1.yaml>)；状态依据：[status.json](</home/pai/Huawei/experiments/FFTGen/P4Transformer_xyz_mocap_100/P4Transformer/20261102_140828/recovery_20261102_oom/status.json>)。
- 最近完整 epoch：100（2026-11-04 05:44:49 +08:00）；训练 MPJPE：19.716 cm；验证 MPJPE：24.917 cm；验证 loss：0.24916680842524877。
- 已记录的最佳验证 MPJPE：13.545 cm（epoch 77）。
- 训练日志：[log.txt](</home/pai/Huawei/experiments/FFTGen/P4Transformer_xyz_mocap_100/P4Transformer/20261102_140828/log/log.txt>)。
- 控制台：[console.log](</home/pai/Huawei/experiments/FFTGen/P4Transformer_xyz_mocap_100/P4Transformer/20261102_140828/recovery_20261102_oom/console.log>)。
- 最近证据更新时间：2026-11-04 05:44:50 +08:00。
- 实验目录：[20261102_140828](</home/pai/Huawei/experiments/FFTGen/P4Transformer_xyz_mocap_100/P4Transformer/20261102_140828>)。
- last.pth：[last.pth](</home/pai/Huawei/experiments/FFTGen/P4Transformer_xyz_mocap_100/P4Transformer/20261102_140828/checkpoint/last.pth>)（更新于 2026-11-04 05:44:49 +08:00）。
- best.pth：[best.pth](</home/pai/Huawei/experiments/FFTGen/P4Transformer_xyz_mocap_100/P4Transformer/20261102_140828/checkpoint/best.pth>)（更新于 2026-11-03 14:55:57 +08:00）。

### FFTGen best第77轮续训30轮（78–107），GPU1

从原FFTGen第77轮best恢复模型、AdamW、77轮指标历史及最佳验证13.545cm；追加30轮至107轮。学习率衔接检查点保存的下一轮值0.0013120001496535435，以30轮余弦衰减至接近零、无预热。保留FFTGen xyz、MOCAP-21、500点、T16、acc3、步长1、训练batch32/验证16、BF16；不新增梯度裁剪。新目录独立保存，原实验与best保留。用户确认改用空闲GPU1立即启动，GPU0 ConvNeXtV2继续训练。

- 状态：已完成；阶段：complete；已完成 107/107 epoch。
- 配置：[config.yaml](</home/pai/Huawei/experiments/FFTGen/P4Transformer_xyz_mocap_100/P4Transformer/20261102_140828/resume_best30_20261104/config/config.yaml>)；状态依据：[status.json](</home/pai/Huawei/experiments/FFTGen/P4Transformer_xyz_mocap_100/P4Transformer/20261102_140828/resume_best30_20261104/status.json>)。
- 最近完整 epoch：107（2026-11-05 06:19:43 +08:00）；训练 MPJPE：11.235 cm；验证 MPJPE：17.157 cm；验证 loss：0.1715690328916785。
- 已记录的最佳验证 MPJPE：13.440 cm（epoch 86）。
- 训练日志：[log.txt](</home/pai/Huawei/experiments/FFTGen/P4Transformer_xyz_mocap_100/P4Transformer/20261102_140828/resume_best30_20261104/log/log.txt>)。
- 控制台：[console.log](</home/pai/Huawei/experiments/FFTGen/P4Transformer_xyz_mocap_100/P4Transformer/20261102_140828/resume_best30_20261104/console.log>)。
- 最近证据更新时间：2026-11-05 06:19:44 +08:00。
- 实验目录：[resume_best30_20261104](</home/pai/Huawei/experiments/FFTGen/P4Transformer_xyz_mocap_100/P4Transformer/20261102_140828/resume_best30_20261104>)。
- last.pth：[last.pth](</home/pai/Huawei/experiments/FFTGen/P4Transformer_xyz_mocap_100/P4Transformer/20261102_140828/resume_best30_20261104/checkpoint/last.pth>)（更新于 2026-11-05 06:19:43 +08:00）。
- best.pth：[best.pth](</home/pai/Huawei/experiments/FFTGen/P4Transformer_xyz_mocap_100/P4Transformer/20261102_140828/resume_best30_20261104/checkpoint/best.pth>)（更新于 2026-11-04 16:49:30 +08:00）。

### FFTGen 原版best第21轮，stride1微调50轮

从20261022_232202第21轮best恢复模型、AdamW和历史；COCO17、xyz+2特征、200点、T16、acc3、stride1、batch64、BF16；lr3e-5、无预热、余弦50轮至71轮。冻结源码与配置，复用当前FFTGen缓存（训练639633/验证150644窗口）；与历史数据划分不同，独立统计best，原始12.935cm仅作历史参考。

- 状态：已完成；阶段：complete；已完成 71/71 epoch。
- 配置：[config.yaml](</home/pai/Huawei/experiments/FFTGen/P4Transformer/20261104_213441/config/config.yaml>)；状态依据：[status.json](</home/pai/Huawei/experiments/FFTGen/P4Transformer/20261104_213441/status.json>)。
- 最近完整 epoch：71（2026-11-05 22:00:52 +08:00）；训练 MPJPE：5.699 cm；验证 MPJPE：13.435 cm；验证 loss：0.13434847616744927。
- 已记录的最佳验证 MPJPE：12.421 cm（epoch 22）。
- 训练日志：[log.txt](</home/pai/Huawei/experiments/FFTGen/P4Transformer/20261104_213441/log/log.txt>)。
- 控制台：[console.log](</home/pai/Huawei/experiments/FFTGen/P4Transformer/20261104_213441/log/console.log>)。
- 最近证据更新时间：2026-11-05 22:00:53 +08:00。
- 实验目录：[20261104_213441](</home/pai/Huawei/experiments/FFTGen/P4Transformer/20261104_213441>)。
- last.pth：[last.pth](</home/pai/Huawei/experiments/FFTGen/P4Transformer/20261104_213441/checkpoint/last.pth>)（更新于 2026-11-05 22:00:52 +08:00）。
- best.pth：[best.pth](</home/pai/Huawei/experiments/FFTGen/P4Transformer/20261104_213441/checkpoint/best.pth>)（更新于 2026-11-04 22:05:51 +08:00）。

### FFTGen 数据处理后重跑：原best stride1微调50轮

从20261022_232202第21轮best恢复模型/AdamW，追加50轮至71；COCO17、xyz+2特征、200点、T16、acc3、batch64、BF16、lr3e-5余弦50轮、无预热。使用2026-11-05冻结代码重新构建数据集，启用人员连续性和GT骨架质量过滤，当前帧裁剪至少10点且整个人窗口完整有效。先用同一新验证集评估初始best作为baseline，新阶段独立统计best；过滤统计保存在config/dataset.json与日志中。

- 状态：运行中；阶段：training；已完成 44/71 epoch。
- 配置：[config.yaml](</home/pai/Huawei/experiments/FFTGen/P4Transformer/20261105_203931/config/config.yaml>)；状态依据：[status.json](</home/pai/Huawei/experiments/FFTGen/P4Transformer/20261105_203931/status.json>)。
- PID：78754；用户：pai；进程开始：2026-11-05 22:46:24 +08:00。
- 最近完整 epoch：44（2026-11-06 14:06:51 +08:00）；训练 MPJPE：7.428 cm；验证 MPJPE：13.176 cm；验证 loss：0.13175759644061139。
- 已记录的最佳验证 MPJPE：13.135 cm（epoch 40）。
- 预计训练结束：2026-11-07 06:46:45 +08:00（最近完整 epoch 耗时中位数，仅供参考，不含后续评估）。
- 训练日志：[log.txt](</home/pai/Huawei/experiments/FFTGen/P4Transformer/20261105_203931/log/log.txt>)。
- 控制台：[console.log](</home/pai/Huawei/experiments/FFTGen/P4Transformer/20261105_203931/log/console.log>)。
- 最近证据更新时间：2026-11-06 14:30:01 +08:00。
- 实验目录：[20261105_203931](</home/pai/Huawei/experiments/FFTGen/P4Transformer/20261105_203931>)。
- last.pth：[last.pth](</home/pai/Huawei/experiments/FFTGen/P4Transformer/20261105_203931/checkpoint/last.pth>)（更新于 2026-11-06 14:06:51 +08:00）。
- best.pth：[best.pth](</home/pai/Huawei/experiments/FFTGen/P4Transformer/20261105_203931/checkpoint/best.pth>)（更新于 2026-11-06 11:38:18 +08:00）。

