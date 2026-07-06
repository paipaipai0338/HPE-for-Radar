# RPM 2.0: RF-based Pose Machines for  Multi-Person 3D Pose Estimation

# 框架拆解
![overall](fig/overall.png)
## Feature Extractor
![Feature Extractor](fig/feature_extractor.png)

### Backbone
预训练的HRNet-18 
- 输入 $H_{input} × W_{input}$
- 输出 $H = \frac{H_{input}} {4}，W =\frac{W_{input}} {4} $

### Detection Branch
多个并行 MLP 或 1D conv
- Center Heatmap Head
    - 作用 输出人体中心位置 $H_{xy}$
    - shape $1 \times H \times W$
    - 作用对象 水平 RF heatmap XOY 平面
    - 监督方式 在每个人的包围框中心构造一个二维 Gaussian peak，形成监督热图

        $$
        b_{GT}^i = (x_1^i, y_1^i, x_2^i, y_2^i), \quad i \in [1, 2, \dots K]
        $$

        对于第 $i$ 个人其中心为：
        $
        c_x^i=\frac{x_1^i+x_2^i}{2},
        \qquad
        c_y^i=\frac{y_1^i+y_2^i}{2}
        $

        映射到步长为 4 的特征图：
        $
        (\tilde{c}_x^i, \tilde{c}_y^i) = \left( \left\lfloor \frac{c_x^i}{4} \right\rfloor, \left\lfloor \frac{c_y^i}{4} \right\rfloor \right)
        $

        GT 中心热图为：

        $
        \hat{H}_{x,y} = \sum_{i=1}^{K} \exp\left(-\frac{(x - \hat{c}^i_x)^2 + (y - \hat{c}^i_y)^2}{\sigma^2}\right)
        $

        $
        \mathcal{L}_{center} = \frac{1}{XY} \sum_{xy} (\hat{H}_{xy} - H_{xy})^2
        $
    - 注释 推理阶段需要通过局部峰值、阈值或 top-M 解码中心。实际人数由有效峰值数量间接获得，但论文没有给出具体的峰值解码策略。

- Center Offest Head
    - 作用 因为图像被降采样所以结果会出现小数误差，所以为图像中的每个位置都添加了一个向量场，修正中心位置在步长为 4 的特征图上取整所产生的亚像素量化误差。
    - shape $2 \times H \times W$
    - 监督方式 稠密输出，稀疏监督

        $
        \hat{o}^i = \left( \frac{c_x^i}{4} - \left\lfloor \frac{c_x^i}{4} \right\rfloor, \frac{c_y^i}{4} - \left\lfloor \frac{c_y^i}{4} \right\rfloor \right)
        $

        $
        \mathcal{L}_{offset} = \frac{1}{K} \sum_{i}^{K} \left\| \hat{o}^i - o^i \right\|_1
        $
- Box Size Head
    - 作用 Center Heatmap Head + Center Offest Head 只能得到中心位置，该模块输出尺寸信息
    - shape $4 \times H \times W$
    - 监督方式 稠密输出，稀疏监督

        $
        \hat{s}^i = \left( \tilde{c}_x^i - x_1^i, \tilde{c}_y^i - y_1^i, x_2^i - \tilde{c}_x^i, y_2^i - \tilde{c}_y^i \right)
        $

        $
        \mathcal{L}_{size} = \frac{1}{K} \sum_{i}^{K} \left\| \hat{s}^i - s^i \right\|_1
        $
- Keypoint Heatmap Head
    - 作用 将 HRNet backbone 特征投影为供后续 网络使用的姿态特征图
    - shape $128 \times H \times W$
    - 监督方式 无独立的显式监督项
## Multi-view Fusion Network

经过 Feature Extractor 后两个方向的特征图可表示为
$
E_{hor} \in \mathbb{R}^{128 \times H \times W}
$
$
E_{ver} \in \mathbb{R}^{128 \times H \times W}
$
随后 RoIAlign 利用检测中心以及检测框进行裁剪（既统一了输出维度，又避免了RoIPool的取整误差）

随后利用 1D conv 和 embedding将 hor 和 ver 投影到一共共同空间中去，用三种方式聚合特征。

## Spatio-Temporal Attention Network
本质上为 transformer，但这里的输入直接断层，莫名其妙多了 J 的维度，假设是通过可学习的 J 个 joint queries 诱导出来的。

![Spatio-Temporal Attention Network](fig\stam.png)

SAM 和 TAM 的本质都是一样的利用 transformer 架构进行前向传播，且会随机进行mask（这点确实有点意思，算是一种数据增强的方法），SAM 和 TAM 二者的本质差异是一个把关键点维度当作序列长度，一个把时间当作序列长度进行自注意力计算，同时还加入了可学习的位置编码。

### SAM
有输入
$$
E^{input} \in \mathbb{R}^{T \times M \times J \times C} \Rightarrow X_{spatial} \in \mathbb{R}^{D \times J \times C} \quad where \quad D = T \cdot M
$$

$$
X_{spatial} = [x_{spatial}^1; x_{spatial}^2; \cdots ; x_{spatial}^D]
$$
随后在 J 这个维度随机进行 mask 模拟丢失数据的情况，并加入embedding，送入transformer
$$
x_{spatial} = [x^1; \cdots ; \text{Mask}(x^i); \cdots ; x^J] + \mathbf{PE}_{sp} \quad where \quad \mathbf{PE}_{sp} \in \mathbb{R}^{J \times C}
$$

### TAM
与 SAM 类似，在此不做赘述