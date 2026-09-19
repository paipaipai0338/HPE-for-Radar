# ActionRec

基于毫米波雷达点云和FFT数据的两条四分类行为识别路线，类别顺序为：

| 索引 | 类别               |
| ---: | ------------------ |
|    0 | 站立`stand`      |
|    1 | 坐/蹲`sit_squat` |
|    2 | 躺`lie`          |
|    3 | 其他`other`      |

## 工程结构

当前点云路线集中在`src/pointcloud/`：`data/`负责数据集、增强、缓存和形态特征，`models/`负责
点云与行为模型，`utils/`负责点云关联、姿态和后处理，`scripts/`负责可视化推理。四类标签、损失和
指标、姿态I/O、坐标变换和时间同步位于`src/common/`，可供两条路线复用；时序编码位于
`src/temporal/`。FFT路线位于`src/fft/`，其中`utils/preprocess.py`负责固件数据解码和谱图计算，
`utils/tools.py`存放投影等辅助工具。FFT的数据准备、缓存、模型、训练和推理说明见
[`../fft/README.md`](../fft/README.md)，配置位于`../fft/config/train.yaml`，结果独立存放在`outputs/fft/`。

点云代码直接从`src.pointcloud`导入，公共组件从`src.common`或`src.temporal`导入。

## 输入与预处理

当前模型使用长度为 $T=16$ 的连续窗口，每帧最多保留 $N=128$ 个关联人体点。高位机点云先转换到低位机坐标系，再根据 3D 姿态进行人体点云关联；关联失败的帧不会进入窗口。

窗口内所有帧使用最后一帧点云的 XYZ 中位数统一平移：

$$
\tilde{\boldsymbol p}_{t,i}=\boldsymbol p_{t,i}-
\mathrm{median}_{j}(\boldsymbol p_{T,j}).
$$

该处理去除绝对位置，同时保留窗口内的相对位移。点数不足时补零，并由 `point_mask` 排除补零点；点数超过 128 时进行采样。

| 输入                 | 形状          | 说明                                                       |
| -------------------- | ------------- | ---------------------------------------------------------- |
| `points`           | $[B,T,N,4]$ | XYZ 和径向速度；当前`use_doppler: false`，网络只使用 XYZ |
| `point_mask`       | $[B,T,N]$   | 有效点掩码                                                 |
| `shape_statistics` | $[B,T,8]$   | 采样和增强前计算的平移不变形态与点数统计                   |
| `label`            | $[B]$       | 窗口最后一帧标签                                           |

### 形态统计

`compute_shape_statistics()`在采样和增强前，使用一帧中全部关联人体点计算 15 维统计。先对点云
中心化：

$$
\boldsymbol q_i=\boldsymbol p_i-\bar{\boldsymbol p},
\qquad
r_i=\sqrt{q_{i,x}^2+q_{i,y}^2}.
$$

设 $Q_p(a)$ 为序列 $a$ 的 $p$ 分位数。15 维特征按代码中的固定顺序为：

| 索引 | 特征             | 计算                              | 含义                                              |
| ---: | ---------------- | --------------------------------- | ------------------------------------------------- |
|    0 | 稳健高度         | $Q_{0.9}(z)-Q_{0.1}(z)$         | 排除少量离群点后的竖直跨度                        |
|    1 | 竖直标准差       | $\mathrm{std}(z)$               | 点云沿竖直方向的离散程度                          |
|    2 | 水平半径中位数   | $Q_{0.5}(r)$                    | 人体点云典型的水平扩散范围                        |
|    3 | 水平半径 P90     | $Q_{0.9}(r)$                    | 人体点云外侧的稳健水平范围                        |
|    4 | XY 次轴尺度      | $\sqrt{\lambda_1^{xy}}$         | XY 平面较窄方向的分布尺度                         |
|    5 | XY 主轴尺度      | $\sqrt{\lambda_2^{xy}}$         | XY 平面较长方向的分布尺度                         |
|    6 | 主轴竖直分量     | $|e_{\max,z}|$                  | 三维最大主轴与竖直方向的对齐程度，越接近 1 越竖直 |
|    7 | 归一化点数       | $\log(1+N_t)/\log(129)$         | 采样前关联人体点的数量和稠密程度                  |
|    8 | TLV1 点比例      | $(N_{TLV1}+1)/(N_{TLV1:7}+2)$   | 平滑后的动点占比                                  |
|    9 | 归一化 TLV1 点数 | $\log(1+N_{TLV1})/\log(129)$    | 动态反射点的证据强度                              |
|   10 | 下部点比例       | $N_{lower}/N_t$                 | 点云落在稳健高度下三分之一的比例                  |
|   11 | 中部点比例       | $N_{middle}/N_t$                | 点云落在稳健高度中三分之一的比例                  |
|   12 | 上部点比例       | $N_{upper}/N_t$                 | 点云落在稳健高度上三分之一的比例                  |
|   13 | 下部相对宽度     | $Q_{0.9}(r_{lower})/Q_{0.9}(r)$ | 下部点云相对整帧水平范围的宽度                    |
|   14 | 上部相对宽度     | $Q_{0.9}(r_{upper})/Q_{0.9}(r)$ | 上部点云相对整帧水平范围的宽度                    |

其中 $\lambda_1^{xy}\leq\lambda_2^{xy}$ 是中心化点云 XY 协方差矩阵的两个特征值；
$e_{\max,z}$ 是三维协方差矩阵最大特征值对应特征向量的 Z 分量。

计算第 10 至 14 维时，先使用稳健高度范围归一化 Z 坐标：

$$
\hat z_i=\mathrm{clip}\left(
\frac{z_i-Q_{0.1}(z)}{Q_{0.9}(z)-Q_{0.1}(z)},0,1
\right),
$$

再按照 $[0,1/3)$、$[1/3,2/3)$、$[2/3,1]$ 划分下、中、上三个区域。稳健高度接近零时，
所有点的归一化高度记为 0.5；若某个区域没有点，或整帧水平参考半径接近零，对应相对宽度记为 0。

当前配置为 `shape_statistic_dim: 8`，Dataset会计算完整15维后截取前8维，因此模型当前实际输入
的是索引0至7。使用10维checkpoint时，额外输入索引8、9的TLV活动特征；索引10至14为垂直分层特征。

## 模型结构

当前配置并行使用 PointNet、两层轻量 EdgeConv 和 8 维统计，共有 266,292 个可训练参数，数据流如下：

```text
XYZ点云 [B,T,N,3]
  ├-> PointNet: 3 -> 32 -> 64 -> 池化 -> 128 --------┐
  └-> EdgeConv: 9 -> 64 -> 131 -> 64 -> 池化 -> 128 ─┴-> 空间融合拼接: 256 -> 128 --┐
                                                                                   ├-> 融合: 160 -> 128
8维形态统计 [B,T,8] -> MLP: 8 -> 16 -> 32 --- --------------------------------------┘
  -> 4 层因果空洞 TCN
  -> 分类头: 128 -> 128 -> 4
  -> 最后一帧分类结果
```

### 点云帧编码

PointNet 分支通过共享逐点 MLP 提取全局形态。EdgeConv 分支在每帧 XYZ 空间查找 8 个近邻，并使用下式聚合局部几何：

$$
\boldsymbol e_{ij}^{(l)}=\phi_l([\boldsymbol h_i^{(l)},\
\boldsymbol h_j^{(l)}-\boldsymbol h_i^{(l)},\boldsymbol p_j-\boldsymbol p_i]),
\qquad
\boldsymbol h_i^{(l+1)}=\max_{j\in\mathcal N_8(i)}\boldsymbol e_{ij}^{(l)}.
$$

近邻只由有效点构成，补零点由 `point_mask` 排除。两个分支分别对有效点集合 $V_t$ 计算：

$$
\boldsymbol\mu_t=\frac{1}{|V_t|}\sum_{i\in V_t}\boldsymbol h_{t,i},
\qquad
\boldsymbol\sigma_t=
\sqrt{\frac{1}{|V_t|}\sum_{i\in V_t}(\boldsymbol h_{t,i}-\boldsymbol\mu_t)^2},
$$

$$
\boldsymbol m_t=\max_{i\in V_t}\boldsymbol h_{t,i}.
$$

三种池化结果拼接后分别得到全局和局部帧特征，两者拼接并映射回 128 维：

$$
\boldsymbol f_t^{pc}=\psi([\boldsymbol f_t^{global},\boldsymbol f_t^{local}]).
$$

8 维形态统计经过独立 MLP 编码，与点云特征拼接并映射到 128 维帧特征。

### 时序编码

TCN 包含 4 个残差因果卷积块，卷积核大小为 2，膨胀率依次为 $1,2,4,8$。单个块可写为：

$$
\boldsymbol H^{(l+1)}=\boldsymbol H^{(l)}+
\mathrm{Dropout}\left(
\mathrm{SiLU}\left(
\mathrm{LN}(\mathrm{Conv}_{d_l}(\boldsymbol H^{(l)}))
\right)\right).
$$

卷积只在左侧补零，因此时刻 $t$ 不会使用未来帧。总感受野为：

$$
R=1+(k-1)\sum_l d_l=1+(2-1)(1+2+4+8)=16.
$$

最后使用第 16 帧的时序特征输出四类 logits。

## 推理后处理

`src/pointcloud/utils/postprocess.py`对按时间顺序排列的窗口执行因果状态平滑，仅用于推理，不参与训练损失。
输入包括模型四类概率、关联点云的TLV1比例、点数和稳健高度。

候选类别直接使用模型概率判断，不重复进行指数平滑。TLV1比例不低于0.25时进入动态状态，
并将快速确认状态延续4帧；动态或近期动态时，候选类别连续2帧成立后切换，静态时连续8帧后切换。
若非`other`候选概率不低于0.8且当前状态概率不高于0.2，也使用2帧确认，以便从错误维持状态中恢复。
静态状态切换到`other`需要连续16帧确认；发生运动时仍使用2帧，避免静止人体点云残缺造成短时误判。

两种情况均要求候选概率不低于0.55，且相比当前状态至少领先 0.08。TLV只控制切换速度，不直接指定行为类别。

为避免残缺点云建立错误状态，使用最近16个可靠帧的点数和稳健高度中位数作为参考。当静态帧同时
满足以下条件时，将其视为低质量帧：

$$
N_t<0.70\mathrm{median}(N),\qquad
H_t<0.80\mathrm{median}(H).
$$

低质量帧不能发起状态切换，也不写回质量基线；动态帧不受该门控限制。同一采集组中暂时缺少有效
输入时保留上一状态，新的目标序列需要显式重置。稳定静态状态每8帧复核一次；复核产生候选类别后
暂停状态维持并连续推理，直到候选得到确认或被后续结果否定。

以下结果来自调整前的后处理版本，仅作为历史基线。使用10维Focal模型在20260709的37组、
29887个stride1窗口上顺序评估：

| 方式     | Accuracy | Macro F1 |
| -------- | -------: | -------: |
| 原始预测 |   0.9673 |   0.9294 |
| 后处理   |   0.9692 |   0.9327 |

后处理在17组增加正确帧、5组不变、15组下降，净增加56个正确窗口。它能减少稳定段毛刺，但会在
部分真实类别切换处引入延迟。当前逻辑仍需在相同验证集上重新评估后再与该结果比较。

## 损失函数

训练集按窗口最后一帧统计类别数量。类别 $c$ 的权重为：

$$
w_c=\frac{N_{train}}{C\,n_c},
$$

其中 $C=4$，$n_c$ 为类别 $c$ 的窗口数。一个 batch 的加权交叉熵为：

$$
\mathcal L_{final}=
-\frac{\sum_i w_{y_i}\log p_{i,y_i}}
{\sum_i w_{y_i}}.
$$

代码保留了逐帧辅助损失接口：

$$
\mathcal L=\frac{\mathcal L_{final}+\lambda\mathcal L_{aux}}{1+\lambda}.
$$

当前配置 `auxiliary_loss_weight: 0`，即 $\lambda=0$，训练只监督窗口最后一帧。

## 评估指标

混淆矩阵 $M$ 的行表示真实类别，列表示预测类别。对类别 $c$：

$$
TP_c=M_{c,c},\qquad
FP_c=\sum_i M_{i,c}-TP_c,\qquad
FN_c=\sum_j M_{c,j}-TP_c.
$$

$$
P_c=\frac{TP_c}{TP_c+FP_c},\qquad
R_c=\frac{TP_c}{TP_c+FN_c},\qquad
F1_c=\frac{2P_cR_c}{P_c+R_c}.
$$

总体指标为：

$$
\mathrm{Accuracy}=\frac{\sum_c M_{c,c}}{\sum_{i,j}M_{i,j}},
\qquad
\mathrm{MacroF1}=\frac{1}{C}\sum_c F1_c.
$$

模型选择以验证集 Macro F1 为准，同时输出每类 Precision、Recall、F1 和完整混淆矩阵。

## 当前训练配置

- 窗长 16，训练和验证窗口步长均为 8。
- 点特征使用 XYZ，径向速度默认关闭。
- 单帧点云编码并行使用 PointNet 和两层 EdgeConv，近邻数为 8；8 维统计保留为独立分支。
- 训练增强：绕 Z 轴旋转 $\pm120^\circ$、绕 X/Y 轴旋转 $\pm3^\circ$、坐标噪声标准差 0.01、随机丢点 0% 至 10%。
- 优化器为 AdamW，学习率 $10^{-4}$，权重衰减 $10^{-5}$。
- `ReduceLROnPlateau` 监控验证集 Macro F1，连续 3 个 epoch 无提升时学习率减半。
- 连续 10 个 epoch 无提升时提前停止，保存 Macro F1 最佳的模型。

完整参数见 [`config/train.yaml`](config/train.yaml)。训练命令：

```bash
python -m src.pointcloud.train
```

点云实验保存在`outputs/pointcloud/runs/`，点云索引和关联缓存保存在
`outputs/pointcloud/cache/`。

实验结果记录在 [`experiment.md`](../../experiment.md)。
