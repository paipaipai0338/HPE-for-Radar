# 训练监控

入口：[training_monitor.md](training_monitor.md)。每逢北京时间整 10 分钟刷新，读取已有训练状态、日志、checkpoint 文件时间和宿主机 GPU / 进程信息；不加载权重，不修改训练，不启动排队任务。

## 文件

| 文件 | 用途 | 是否保留 |
| --- | --- | --- |
| `refresh.py` | 仅使用 Python 标准库采集训练状态并生成快照 | 核心，保留 |
| `tasks.json` | 任务登记、实验目的、配置位置、状态来源和启动条件 | 核心，保留；顺序只用于展示 |
| `monitor.cron` | 每 10 分钟刷新的定时配置源文件 | 保留；实际执行由用户 crontab 注册项负责 |
| `bark_monitor.py` | 独立监控训练完成 / 失败并发送 Bark 通知 | 使用手机通知时保留 |
| `test_refresh.py` / `test_bark_monitor.py` | 监控解析、状态判断、失败保护及通知去重测试 | 维护检查用，建议保留 |
| `training_monitor.md` | 自动覆盖的最新快照；人工说明请改 `tasks.json` | 自动生成，保留便于查看 |
| `health.json` | 最近一次采集结果；整体失败时保留上一份快照 | 自动生成，保留便于排错 |
| `monitor.log` | 刷新日志和异常堆栈 | 自动追加，需要时可清空旧内容 |
| `refresh.lock` | 防止并发刷新 | 运行时保留；空文件正常，运行中不要删除 |

训练原始配置、日志和模型留在原位置，以链接引用。系统 cron 保存一份定时注册记录。Bark 独立使用项目根目录的 `.env.bark` 读取推送地址，在 `experiments/.bark-monitor/` 保存去重状态、锁及运行日志；保留去重状态，避免重启后重复通知。

## 操作

在宿主机手动刷新（沙箱无法读取真实进程和 GPU，脚本会拒绝覆盖快照）：

```bash
/usr/bin/python3 -B /home/pai/Huawei/supervise/refresh.py
```

检查定时任务和采集结果：

```bash
crontab -l
cat /home/pai/Huawei/supervise/health.json
tail -n 10 /home/pai/Huawei/supervise/monitor.log
```

新增任务：复制 `tasks.json` 中相似记录，填写唯一 `id`、任务目的、`state` 状态文件、`process_match` 进程命令片段、配置及日志位置。状态文件使用现有 `status`、PID、`experiment`、`stage` 等字段。训练出现新任务时需登记其目的和状态来源；GPU 中未登记进程仍会显示占用，但不会猜测其配置。没有进程匹配时显示“待核实”，不会直接推断为完成。

串行队列中的模型分别登记自己的状态文件；`queue_state` 和 `queue_match` 指向父队列状态与进程，用于确认 `queued` 模型仍在排队。父队列失败时，未启动模型显示“队列已停止”。续训追加同一份日志时，可用 `log_after` 指定本阶段标记，`log_before` 保留历史阶段；目标轮数优先读取状态中的 `target_epochs`，避免把旧指标和轮数算进新阶段。

运行检查：

```bash
/usr/bin/python3 -B -m unittest discover -s /home/pai/Huawei/supervise -p 'test_*.py'
```

停止自动刷新：用 `crontab -e` 删除含 `/home/pai/Huawei/supervise/refresh.py` 的一行，保留其他定时任务。

## 判断范围

epoch 和最佳 MPJPE 来自实验日志中的完整 epoch 记录；状态文件中的当前阶段用于区分训练、验证和对比评估。“当前任务”表与详情共用预计结束时间，最近最多取 11 条完整 epoch 日志，有多少用多少，以有效相邻轮次耗时的中位数估计；只有一条日志时，用进程启动至该轮完成的耗时（包含初始化）估计。无有效耗时数据或正在评估时显示“暂无法估算”。使用北京时间，不含后续评估；单条日志、中断恢复和其他用户共享 GPU 会影响估计。当前配置文件可能被修改，详情应以实验目录内保存的配置为准。

任务仍在运行但状态及日志超过 30 分钟不更新时提示检查，不自动认定卡死。异常发现可能延迟约 10 分钟。历史任务仅展示已登记项，未登记的计划不能自动推断。监控日志每次刷新追加一行，脚本失败时追加异常堆栈。
