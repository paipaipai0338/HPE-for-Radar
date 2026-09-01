# HRRadarPose 《HRRadarPose: A 4D Radar Tensor-Based 3D Human Pose Estimation》

## 数据处理
```
In this study, the processed 4D radar tensor has the dimension of 64×32×128×256, which correspond to velocity, the z-axis, the y-axis, and the x-axis, respectively.
```
单帧的信号被处理成为 R $\in \mathbb{R}^{Range\times Doppler\times Azi\times Ele}$ ，随后将空间体素化，每个体素 $(x_i, y_i, z_i)$可以按照空间位置找到 $R_i, A_i, E_i$的索引进而可以生成结果 T $\in \mathbb{R}^{D\times X\times Y\times Z}$

## 输出头
为了避免显式编码人数，这里直接用相同shape的体素化结果作为输出。
具体来说 最高分辨率的 HRNet 输出特征后面接入了两个卷积分支 **Body Center Probability** 和 **Keypoint Offset**，假定输入维度 $X\in \mathbb{R}^{B \times D \times X_h \times Y_h \times Z_h}$，那么**Body Center Probability**输出 $C_s\in \mathbb{R}^{B \times 1 \times X_h \times Y_h \times Z_h}$，**Keypoint Offset**输出 $K_s\in \mathbb{R}^{B \times J*3 \times X_h \times Y_h \times Z_h}$

$C_s$通过topK、NMS以及threshold来筛选人数，作为置信度，$K_s$则是基于$C_s$来同步查询组成人体结果。

**Keypoint Offset**在训练过程中通过encoder将其编码为稀疏的形式，利用真值的indices进行监督，训练过程中并非NMS+topK，这里与RPM思想一致
