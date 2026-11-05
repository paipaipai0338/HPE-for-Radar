"""The existing P4 encoder and joint-memory decoder, with gradients across both."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class P4JointFineTune(nn.Module):
    def __init__(self, frontend, decoder, encoder_batch_size=64, checkpoint_encoder=True):
        super().__init__()
        if type(encoder_batch_size) is not int or encoder_batch_size < 1:
            raise ValueError('encoder_batch_size must be a positive integer')
        if type(checkpoint_encoder) is not bool:
            raise ValueError('checkpoint_encoder must be bool')
        self.tube_embedding = frontend.tube_embedding
        self.pos_embedding = frontend.pos_embedding
        self.transformer = frontend.transformer
        self.emb_relu = frontend.emb_relu
        self.feat_dim = frontend.feat_dim
        self.decoder = decoder
        self.encoder_batch_size = encoder_batch_size
        self.checkpoint_encoder = checkpoint_encoder

    def set_encoder_stage(self, train_points):
        self.transformer.requires_grad_(True)
        self.pos_embedding.requires_grad_(True)
        self.tube_embedding.requires_grad_(train_points)

    def _encode_windows(self, selected):
        # The pointnet2 extension launches on the current device.
        with torch.cuda.device_of(selected):
            positions, features = self.tube_embedding(selected[..., :3].contiguous(),
                selected[..., 3:].permute(0, 1, 3, 2).contiguous())
            b, t, n, _ = positions.shape
            time = torch.arange(1, t+1, device=selected.device, dtype=positions.dtype)
            xyzt = torch.cat((positions, time[None, :, None, None].expand(b, t, n, 1)), -1)
            embedded = features.permute(0, 1, 3, 2).reshape(b, t*n, -1)
            embedded = embedded+self.pos_embedding(xyzt.reshape(b, t*n, 4).transpose(1, 2)).transpose(1, 2)
            if self.emb_relu:
                embedded = self.emb_relu(embedded)
            last = self.transformer(embedded).reshape(b, t, n, -1)[:, -1]
            return last, positions[:, -1]

    def encode(self, batch):
        points = batch['points']
        valid = batch['valid']
        if (points.ndim != 5 or valid.shape != points.shape[:2] or valid.dtype != torch.bool
                or points.shape[-1] != 3+self.feat_dim or points.dtype != torch.float32):
            raise ValueError('Expected FP32 points [B,L,T,N,3+F] and bool valid [B,L]')
        selected = points[valid]
        anchors = points.shape[-2]//self.tube_embedding.spatial_stride
        feature_dim = self.decoder.feature_dim
        xyz = points.new_zeros(*valid.shape, anchors, 3)
        tokens = points.new_zeros(*valid.shape, anchors, feature_dim)
        pooled = points.new_zeros(*valid.shape, feature_dim)
        if len(selected):
            if not torch.isfinite(selected).all():
                raise ValueError('Encoder point windows must be finite')
            outputs = []
            for windows in selected.split(self.encoder_batch_size):
                if self.training and torch.is_grad_enabled() and self.checkpoint_encoder:
                    outputs.append(checkpoint(self._encode_windows, windows, use_reentrant=False))
                else:
                    outputs.append(self._encode_windows(windows))
            last = torch.cat([v[0] for v in outputs])
            positions = torch.cat([v[1] for v in outputs])
            # Preserve FP32 geometry even while the encoder uses BF16 autocast.
            with torch.autocast(points.device.type, enabled=False):
                xyz[valid] = positions.float() @ batch['rotation'][valid].float().transpose(-1, -2)+batch['translation'][valid].float()[:, None, :]
            tokens[valid] = last.float()
            pooled[valid] = last.max(1).values.float()
        return dict(tokens=tokens, xyz=xyz, features=pooled, quality=batch['quality'],
                    rotation=batch['rotation'], translation=batch['translation'],
                    token_mask=valid[..., None].expand(*valid.shape, anchors))

    def forward(self, batch, state=None):
        return self.decoder(self.encode(batch), state)
