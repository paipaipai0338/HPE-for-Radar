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
0.训练采用BF16精度 记录二
1.数据预处理将所有数据放入正数范围 记录三
2.网络通道能力逐级递减，加入group卷积 记录四
3.未移除静态目标 ```remove_static=False``` 记录五
4.雷达信号处理 doppler 维度数量消融：选取中间部分通道、FFT点数少一些

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

原因：经过核查大多数通道维度数据都小于0 经过卷积后Relu可能存在部分数据丢失，
虽然卷积参数可能存在权重为负数但原有整数可能会被妥协到负数，此外卷积带有bias但是完全依赖第一层卷积的bias不如直接对数据进行预处理

[实验目录](/experiments/HRRadarPose/20260902_113015/log/log.txt)

数据 随机挑选

划分方式 group 

效果展示 训练轮数不是很充足 时间不充裕，无法从训练日志中获取结果，但是从原理上来说可行

结论 无明确结论


## 记录四
方法 HRRadarPose

目的：增强网络通道建模能力

与记录三区别：通道增强，group=4
```
stage2_inplanes: 64
stage2_num_channels: [64, 64]
stage3_num_channels: [64, 256, 512]
stage4_num_channels: [64, 256, 512, 1024]
```

[实验目录](/experiments/HRRadarPose/20260902_122404/log/log.txt)

数据 随机挑选

划分方式 group 

效果展示
```
epoch 4 train={'hrradarpose_body_center': 1.6513899187774659, 'hrradarpose_keypoint_offset': 2.0877289389433544} | val={'hrradarpose_body_center': 2.880974227493761, 'hrradarpose_keypoint_offset': 3.217063099340464, 'loss': 6.098037326834225}
```


结论 
训练集下降更快，收敛更快说明增强通道确实对特征提取可行，通道建模能力增强有效

## 记录五
方法 HRRadarPose

目的：检验是否是环境杂波过度干扰模型训练

与记录四区别：remove_static=True

[实验目录](/experiments/HRRadarPose/20260902_150044/log/log.txt)

数据 随机挑选

划分方式 group 

效果展示

结论 
理论有效 暂无结论

## 记录六
方法 HRRadarPose

目的：检验不做数值归一化是否有效

与记录五区别：删除 torch.log().clamp_min(-10)+10 采用原始功率

[实验目录](/experiments/HRRadarPose/20260902_210814/log/log.txt)

数据 随机挑选

划分方式 group 

效果展示

结论 
理论有效 暂无结论

# 最终过拟合结论 
雷达bin中出现了部分坏帧导致val出现了异常，上述分析均无效，且由于训练输出过慢 改用 ResNet3D 继续训练。