import torch
from torch import nn
import numpy as np
from typing import *

from models.P4Transformer import pointnet2_utils
from models.P4Transformer.transformer_helper import Transformer

class P4DConv(nn.Module):
    def __init__(self,
                 in_planes: int,
                 mlp_planes: List[int],
                 mlp_batch_norm: List[bool],
                 mlp_activation: List[bool],
                 spatial_kernel_size: [float, int],
                 spatial_stride: int,
                 temporal_kernel_size: int,
                 temporal_stride: int = 1,
                 temporal_padding: [int, int] = [0, 0],
                 temporal_padding_mode: str = 'replicate',
                 operator: str = '+',
                 spatial_pooling: str = 'max',
                 temporal_pooling: str = 'sum',
                 bias: bool = False):
        """_summary_

        Args:
            in_planes (int): 输入数据中特征维度
            mlp_planes (List[int]): 点云xyz与特征逐步升维 维度list
            mlp_batch_norm (List[bool]): 升维过程中是否加入归一化
            mlp_activation (List[bool]): 升维过程中是否加入非线性激活函数
            spatial_kernel_size (float, int]): 空间卷积的半径 r 以及邻域数量 k 
            spatial_stride (int): 点云数量降采样倍数
            temporal_kernel_size (int): 时间窗口大小
            temporal_stride (int, optional): 时间维度步长. Defaults to 1.
            temporal_padding (int, int], optional): 时间维度padding数量. Defaults to [0, 0].
            temporal_padding_mode (str, optional): 时间维度padding模式. Defaults to 'replicate'.
            operator (str, optional): xyzt 与 特征融合方式. Defaults to '+'.
            spatial_pooling (str, optional): 空间邻域特征选择方式 max mean sum. Defaults to 'max'.
            temporal_pooling (str, optional): 时间邻域特征选择方式. Defaults to 'sum'.
            bias (bool, optional): 卷积 mlp 是否具有bias. Defaults to False.
        """
        super().__init__()

        self.in_planes = in_planes
        self.mlp_planes = mlp_planes
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_activation = mlp_activation

        self.r, self.k = spatial_kernel_size
        self.spatial_stride = spatial_stride

        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_stride = temporal_stride
        self.temporal_padding = temporal_padding
        self.temporal_padding_mode = temporal_padding_mode

        self.operator = operator
        self.spatial_pooling = spatial_pooling
        self.temporal_pooling = temporal_pooling

        # 输入通道 4: xyzt 升维 到 mlp_planes[0] 并加入归一化与非线性激活函数
        conv_d = [nn.Conv2d(in_channels=4, out_channels=mlp_planes[0], kernel_size=1, stride=1, padding=0, bias=bias)]
        if mlp_batch_norm[0]:
            conv_d.append(nn.BatchNorm2d(num_features=mlp_planes[0]))
        if mlp_activation[0]:
            conv_d.append(nn.ReLU(inplace=True))
        self.conv_d = nn.Sequential(*conv_d)
        # 输入通道 in_planes: 原始特征维度升维 到 mlp_planes[0] 并加入归一化与非线性激活函数
        if in_planes != 0:
            conv_f = [nn.Conv2d(in_channels=in_planes, out_channels=mlp_planes[0], kernel_size=1, stride=1, padding=0, bias=bias)]
            if mlp_batch_norm[0]:
                conv_f.append(nn.BatchNorm2d(num_features=mlp_planes[0]))
            if mlp_activation[0]:
                conv_f.append(nn.ReLU(inplace=True))
            self.conv_f = nn.Sequential(*conv_f)

        mlp = []
        for i in range(1, len(mlp_planes)):
            if mlp_planes[i] != 0:
                mlp.append(nn.Conv2d(in_channels=mlp_planes[i-1], out_channels=mlp_planes[i], kernel_size=1, stride=1, padding=0, bias=bias))
            if mlp_batch_norm[i]:
                mlp.append(nn.BatchNorm2d(num_features=mlp_planes[i]))
            if mlp_activation[i]:
                mlp.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*mlp)


    def forward(self, xyzs: torch.Tensor, features: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyzs: torch.Tensor
                 (B, T, N, 3) tensor of sequence of the xyz coordinates
            features: torch.Tensor
                 (B, T, C, N) tensor of sequence of the features
        """
        device = xyzs.device

        nframes = xyzs.size(1)
        npoints = xyzs.size(2)

        assert (self.temporal_kernel_size % 2 == 1), "P4DConv: 时间卷积核大小应该为奇数!"
        assert ((nframes + sum(self.temporal_padding) - self.temporal_kernel_size) % self.temporal_stride == 0), "P4DConv: Temporal length error!"

        # 将 (B, T, N, 3) 转化为 list 长度为 T 元素为 (B, N, 3) 
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]

        if self.temporal_padding_mode == 'zeros':
            # 零填充 前后padding
            xyz_padding = torch.zeros(xyzs[0].size(), dtype=torch.float32, device=device)
            for i in range(self.temporal_padding[0]):
                xyzs = [xyz_padding] + xyzs
            for i in range(self.temporal_padding[1]):
                xyzs = xyzs + [xyz_padding]
        else:
            # 复制填充
            for i in range(self.temporal_padding[0]):
                xyzs = [xyzs[0]] + xyzs
            for i in range(self.temporal_padding[1]):
                xyzs = xyzs + [xyzs[-1]]

        if self.in_planes != 0:
            # 将 (B, T, C, N) 转化为 list 长度为 T 元素为 (B, C, N) 
            features = torch.split(tensor=features, split_size_or_sections=1, dim=1)
            features = [torch.squeeze(input=feature, dim=1).contiguous() for feature in features]
            # 同样进行填充
            if self.temporal_padding_mode == 'zeros':
                feature_padding = torch.zeros(features[0].size(), dtype=torch.float32, device=device)
                for i in range(self.temporal_padding[0]):
                    features = [feature_padding] + features
                for i in range(self.temporal_padding[1]):
                    features = features + [feature_padding]
            else:
                for i in range(self.temporal_padding[0]):
                    features = [features[0]] + features
                for i in range(self.temporal_padding[1]):
                    features = features + [features[-1]]

        new_xyzs = []
        new_features = []
        for t in range(self.temporal_kernel_size//2, len(xyzs)-self.temporal_kernel_size//2, self.temporal_stride):                 # temporal anchor frames
            # t 是时间序列上的采样 只选取每个时间卷积窗口的尺寸中心位置，这也是为什么要保证卷积核尺寸为奇数的原因
            # 对每帧数据利用 pointnet2 进行最远点采样获取具有代表性的 npoints//self.spatial_stride 个点的 id
            # 最远点采样 FPS 保证选出的点均匀分布在整个点云空间
            anchor_idx = pointnet2_utils.furthest_point_sample(xyzs[t], npoints//self.spatial_stride)                               # (B, N//self.spatial_stride)
            anchor_xyz_flipped = pointnet2_utils.gather_operation(xyzs[t].transpose(1, 2).contiguous(), anchor_idx)                 # (B, 3, N//self.spatial_stride)
            anchor_xyz_expanded = torch.unsqueeze(anchor_xyz_flipped, 3)                                                            # (B, 3, N//spatial_stride, 1)
            anchor_xyz = anchor_xyz_flipped.transpose(1, 2).contiguous()                                                            # (B, N//spatial_stride, 3)

            new_feature = []
            for i in range(t-self.temporal_kernel_size//2, t+self.temporal_kernel_size//2+1):
                # 取整个时间卷积核内的数据
                neighbor_xyz = xyzs[i]      # (B, N, 3)
                # 在半径为 self.r 的空间中找到最近的 k 个数据的索引，输出一定是 k 多了截断少了复制
                idx = pointnet2_utils.ball_query(self.r, self.k, neighbor_xyz, anchor_xyz)                                          # (B, N//spatial_stride, self.k)

                neighbor_xyz_flipped = neighbor_xyz.transpose(1, 2).contiguous()                                                    # (B, 3, N)
                # 按照索引形成组
                neighbor_xyz_grouped = pointnet2_utils.grouping_operation(neighbor_xyz_flipped, idx)                                # (B, 3, N//spatial_stride, k)

                # 计算邻居点相对于中心锚点的相对位置坐标
                xyz_displacement = neighbor_xyz_grouped - anchor_xyz_expanded                                                       # (B, 3, N//spatial_stride, k)
                # 获取当前帧相对于当前锚点时刻的时间差
                t_displacement = torch.ones((xyz_displacement.size()[0], 1, xyz_displacement.size()[2], xyz_displacement.size()[3]), dtype=torch.float32, device=device) * (i-t)
                displacement = torch.cat(tensors=(xyz_displacement, t_displacement), dim=1, out=None)                               # (B, 4, N//spatial_stride, k)
                # xyzt 升维
                displacement = self.conv_d(displacement)

                
                if self.in_planes != 0:
                    # 按照索引形成组
                    neighbor_feature_grouped = pointnet2_utils.grouping_operation(features[i], idx)                                 # (B, in_planes, N//spatial_stride, k)
                    # 特征升维
                    feature = self.conv_f(neighbor_feature_grouped)                                                                 # (B, mlp_planes[0], N//spatial_stride, k)
                    if self.operator == '+':
                        feature = feature + displacement
                    else:
                        feature = feature * displacement
                else:
                    feature = displacement
                # 数据逐步升维
                feature = self.mlp(feature)                                                                                         # (B, mlp_planes[-1], N//spatial_stride, k)
                if self.spatial_pooling == 'max':
                    feature = torch.max(input=feature, dim=-1, keepdim=False)[0]                                                    # (B, mlp_planes[-1], N//spatial_stride)
                elif self.spatial_pooling == 'sum':
                    feature = torch.sum(input=feature, dim=-1, keepdim=False)
                else:
                    feature = torch.mean(input=feature, dim=-1, keepdim=False)
                # 单个时序窗口内容处理完成后 append 成为 list
                new_feature.append(feature)
            # 堆叠时间维度
            new_feature = torch.stack(tensors=new_feature, dim=1)                                                                   # (B, T, mlp_planes[-1], N//spatial_stride)

            if self.temporal_pooling == 'max':
                new_feature = torch.max(input=new_feature, dim=1, keepdim=False)[0]                                                 # (B, mlp_planes[-1], N//spatial_stride)
            elif self.temporal_pooling == 'sum':
                new_feature = torch.sum(input=new_feature, dim=1, keepdim=False)
            else:
                new_feature = torch.mean(input=new_feature, dim=1, keepdim=False)
            new_xyzs.append(anchor_xyz)
            new_features.append(new_feature)
        # 重组数据 (B, T, N//spatial_stride, 3) (B, T, mlp_planes[-1], N//spatial_stride)
        new_xyzs = torch.stack(tensors=new_xyzs, dim=1)
        new_features = torch.stack(tensors=new_features, dim=1)

        return new_xyzs, new_features

class P4Transformer(nn.Module):
    r"""
    P4Transformer implementation of human pose estimation (hpe), for dataset "mmBody Benchmark: 3D Body Reconstruction Dataset and Analysis for Millimeter Wave Radarr".

    args:
        radius (float): Param for Point 4D convolution. Radius of point anchor for point grouping. Default 0.1.

        nsamples (int): Param for Point 4D convolution. Number of points for each point anchor. Default 32.

        spatial_stride (int): Param for Point 4D convolution. Spatial stride for each point anchor. Default 3.

        temporal_kernel_size (int): Param for Point 4D convolution. Temporal window size for each frame anchor. Default 3.

        temporal_stride (int): Param for Point 4D convolution. Temporal stride for each frame anchor. Default 2.

        emb_relu (int): Whether using relu embedding. Default False.

        dim (int): Param for Transformer. Feature dimension of transformerf. Default 1024.

        depth (int): Param for Transformer. Depth of transformer. Default 10.

        heads (int): Param for Transformer. Heads of transformer. Default 8.

        dim_head (int): Param for MLP head. Feature dimension of MLP head. Default 256.

        num_classes (int): Output dimension. Default 17*3.

        dropout1 (float): Dropout rate between [0, 1] for Transformer.

        dropout2 (float): Dropout rate between [0, 1] for MLP head.


    """
    def __init__(self, 
                 num_joints=17, max_people=4, feat_dim=2,                                                   # data
                 radius=0.1, nsamples=32, spatial_stride=32,
                 temporal_kernel_size=3, temporal_stride=1,
                 emb_relu=False,
                 dim=1024, depth=10, heads=8, dim_head=256,
                 mlp_dim=2048, dropout1=0.0, dropout2=0.0,                                            # dropout
                ):                                           
        super().__init__()

        self.tube_embedding = P4DConv(in_planes=feat_dim, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 1],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.emb_relu = nn.ReLU() if emb_relu else False

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout1)

        self.mlp_head_pose = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            # nn.Dropout(dropout2),
            nn.Linear(mlp_dim, max_people*num_joints*3),
        )
        self.mlp_head_confidence = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            # nn.Dropout(dropout2),
            nn.Linear(mlp_dim, max_people),
            nn.Sigmoid(),
        )

        self.num_joints = num_joints
        self.feat_dim = feat_dim
        self.max_people = max_people

        
    def forward(self, model_input): 
        f'''
        The model receive aggregated point clouds of 4 frames (referring to mmBody dataset), and output the 3d coordinate of n_p (n_p = 17) joints/keypoints.

        Input:
            Aggregated point clouds according to mmBody dataset, with preprocessing: (1) aggregation (n_frame = 4, stride = 1) (2) padding (n_points = 5000)

        Output:
            3d coordinates of n_p keypoints/joints (tensor): Of shape (b, num_classes//3, 3).

        args:
            model_input:
                input: Aggregated point clouds of shape (B, T, N, D)
                mask: Valid point mask of shape (B, T, N)

        '''
        points = model_input['input']
        mask = model_input['mask']
        points = points * mask.to(dtype=points.dtype).unsqueeze(-1)
        B, T, N, D = points.shape
        point_cloud = points[:, :, :, :3].contiguous()
        point_fea = points[:, :, :, 3:].contiguous()
        point_fea = point_fea.permute(0, 1, 3, 2).contiguous()# [B, L, N, 3]
        device = points.device
        
        assert point_cloud.ndim == 4 and point_cloud.shape[-1] == 3, point_cloud.shape
        assert point_cloud.dtype == torch.float32 and point_cloud.is_contiguous()
        assert point_fea.shape[2] == self.feat_dim and point_fea.is_contiguous()
        assert torch.isfinite(point_cloud).all()

        xyzs, features = self.tube_embedding(point_cloud, point_fea)                                                                                         # [B, L, n, 3], [B, L, C, n] 

        xyzts = []
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]
        for t, xyz in enumerate(xyzs):
            t = torch.ones((xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=device) * (t+1)
            xyzt = torch.cat(tensors=(xyz, t), dim=2)
            xyzts.append(xyzt)
        xyzts = torch.stack(tensors=xyzts, dim=1)
        xyzts = torch.reshape(input=xyzts, shape=(xyzts.shape[0], xyzts.shape[1]*xyzts.shape[2], xyzts.shape[3]))                           # [B, L*n, 4]

        features = features.permute(0, 1, 3, 2)                                                                                             # [B, L, n, C]
        n = features.shape[2]
        features = torch.reshape(input=features, shape=(features.shape[0], features.shape[1]*features.shape[2], features.shape[3]))         # [B, L*n, C]
        
        xyzts_embd = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)

        embedding = xyzts_embd + features
        

        if self.emb_relu:
            embedding = self.emb_relu(embedding)


        # For max-pooling algorithm
        output = self.transformer(embedding)
        output = output.reshape(B, T, n, -1)
        output = torch.max(input=output, dim=2, keepdim=False, out=None)[0]
        
        pose = self.mlp_head_pose(output).view(B, T, self.max_people, self.num_joints, 3)
        confidence = self.mlp_head_confidence(output).view(B, T, self.max_people)

        pre = {
            'pose': pose,
            'confidence': confidence,
        }

        return pre
    
    def get_positional_embeddings1(self, sequence_length, d):
        result = np.ones([1, sequence_length, d])
        for i in range(sequence_length):
            for j in range(d):
                result[0][i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
        return result


if __name__ == "__main__":
    from models.utils.profile_utils import profile_model
    from run.utils.build_model import build_model
    from run.utils.set_device import set_device

    device = set_device(0)
    model = build_model('P4Transformer').to(device)
    x = {
        'input': torch.zeros((1, 1, 300, 6), device=device),
        'mask': torch.ones((1, 1, 300), dtype=torch.bool, device=device),
    }
    profile_model("P4Transformer", model, x)
