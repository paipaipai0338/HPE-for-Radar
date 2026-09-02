# HRRadarPose
## 记录一
方法 HRRadarPose

[实验目录](/experiments/HRRadarPose/20260901_150207/log/log.txt)

数据 随机挑选

划分方式 group 

效果展示
```
train={'hrradarpose_body_center': 0.4832093366243839, 'hrradarpose_keypoint_offset': 1.2374696939942342} | val={'hrradarpose_body_center': 4.519065710906159, 'hrradarpose_keypoint_offset': 3.3315301662365684, 'loss': 7.8505958771427276}
```

原因分析：
训练出现了过拟合，怀疑是64通道的多普勒直接降维度到8网络参数较小能力差
```
FLOPs = 176103.530496 M
MACs = 88051.765248 M
Params = 0.347724 M
```
参数量确实较低，
此外使用FP32进行训练，在当前数据范围情况下epoch一轮 9min
未来工作：
1.全量训练
2.雷达信号处理 doppler 维度数量消融：选取中间部分通道、FFT点数少一些
3.网络通道能力逐级递减，加入group卷积
4.未移除静态目标 ```remove_static=False```
5.调整降低完整epoch数量，加快学习率下降速度

## 记录二
方法 HRRadarPose

目的：检查BF16训练速度与FP32训练速度对比

与记录一区别：采用BF16的精度

[实验目录](/experiments/HRRadarPose/20260902_095656/log/log.txt)

数据 随机挑选

划分方式 group 

效果展示
使用FP32进行训练，在当前数据范围情况下epoch一轮 9min，BF16训练则是 7min，此外datset中步长为4，后续可能改为T

结论 
BF16适用于当前数据

## 记录三
方法 HRRadarPose

目的：检查是否是输入数据数值以及relu的问题

与记录二区别：wrapper结果 clamp_min(-10) + 10

[实验目录](/experiments/HRRadarPose/xxx/log/log.txt)

数据 随机挑选

划分方式 group 

效果展示

结论 

----------------------------------------------------------------------------------------------------------
## 记录
方法 HRRadarPose

目的：增强网络通道建模能力

与记录二区别：通道从64增强到128，并采用分组卷积的操作

[实验目录](/experiments/HRRadarPose/xxx/log/log.txt)

数据 随机挑选

划分方式 group 

效果展示

结论 

## 记录
方法 HRRadarPose

目的：数据角度处理，采用均值对消的方法

与记录三区别：remove_static=True

[实验目录](/experiments/HRRadarPose/xxx/log/log.txt)

数据 随机挑选

划分方式 group 

效果展示

结论 
