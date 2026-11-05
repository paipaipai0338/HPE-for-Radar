"""Joint-specific point-token decoding with causal semantic memory, before XYZ output."""
import torch
from torch import nn
from torch.nn import functional as F


class P4JointMemory(nn.Module):
    def __init__(self, feature_dim=128, hidden_dim=64, num_joints=17, heads=4, head_dim=256, reliability_gating=False):
        super().__init__()
        if any(type(v) is not int or v < 1 for v in (feature_dim, hidden_dim, num_joints, heads, head_dim)):
            raise ValueError('Model dimensions must be positive integers')
        if num_joints != 17 or feature_dim % heads:
            raise ValueError('Require COCO17, positive dimensions and feature_dim divisible by heads')
        self.feature_dim, self.hidden_dim, self.num_joints = feature_dim, hidden_dim, num_joints
        if type(reliability_gating) is not bool:
            raise ValueError("reliability_gating must be bool")
        self.reliability_gating = reliability_gating
        if reliability_gating:
            self.context_gate = nn.Linear(feature_dim+6, feature_dim)
            self.joint_gate = nn.Linear(feature_dim+hidden_dim+16, hidden_dim)
            for gate in (self.context_gate, self.joint_gate):
                nn.init.zeros_(gate.weight)
                nn.init.constant_(gate.bias, 2.)
        self.joint_queries = nn.Parameter(torch.randn(num_joints, feature_dim)*.02)
        self.token_norm = nn.LayerNorm(feature_dim)
        self.position = nn.Sequential(nn.Linear(3, feature_dim), nn.GELU(), nn.Linear(feature_dim, feature_dim))
        self.context_query = nn.Linear(feature_dim, feature_dim)
        self.memory_query = nn.Linear(hidden_dim, feature_dim)
        self.attention = nn.MultiheadAttention(feature_dim, heads, batch_first=True)
        self.quality = nn.Sequential(nn.Linear(6, 16), nn.GELU())
        self.memory = nn.GRUCell(feature_dim+16, hidden_dim)
        self.feature_update = nn.Linear(hidden_dim, feature_dim)
        nn.init.zeros_(self.feature_update.weight)
        nn.init.zeros_(self.feature_update.bias)
        # Each joint starts with its own three rows of the pretrained P4 coordinate head.
        self.head_norm = nn.LayerNorm(feature_dim)
        self.head_hidden = nn.Linear(feature_dim, head_dim)
        self.joint_weight = nn.Parameter(torch.empty(num_joints, 3, head_dim))
        self.joint_bias = nn.Parameter(torch.zeros(num_joints, 3))
        nn.init.xavier_uniform_(self.joint_weight.flatten(0, 1))

    def load_p4_head(self, state):
        with torch.no_grad():
            for name, module in (('0', self.head_norm), ('1', self.head_hidden)):
                module.load_state_dict({key: state[f'mlp_head_pose.{name}.{key}'] for key in ('weight', 'bias')})
            self.joint_weight.copy_(state['mlp_head_pose.3.weight'].reshape_as(self.joint_weight))
            self.joint_bias.copy_(state['mlp_head_pose.3.bias'].reshape_as(self.joint_bias))

    def forward(self, inputs, state=None):
        tokens, xyz, features = (inputs[k].float() for k in ('tokens', 'xyz', 'features'))
        quality, rotation, translation = (inputs[k].float() for k in ('quality', 'rotation', 'translation'))
        if (tokens.ndim != 4 or min(tokens.shape[:3]) < 1 or tokens.shape[-1] != self.feature_dim
                or xyz.shape != (*tokens.shape[:3], 3)
                or features.shape != (*tokens.shape[:2], self.feature_dim)
                or quality.shape != (*tokens.shape[:2], 6)
                or rotation.shape != (*tokens.shape[:2], 3, 3) or translation.shape != (*tokens.shape[:2], 3)):
            raise ValueError('Expected point tokens [B,T,N,F], XYZ [B,T,N,3] and aligned frame fields')
        mask = inputs.get('token_mask', torch.ones(tokens.shape[:3], device=tokens.device, dtype=torch.bool))
        if mask.shape != tokens.shape[:3] or mask.dtype != torch.bool:
            raise ValueError('token_mask must be bool [B,T,N]')
        tokens = tokens.masked_fill(~mask[..., None], 0.)
        xyz = xyz.masked_fill(~mask[..., None], 0.)
        if not all(torch.isfinite(v).all() for v in (tokens, xyz, features, quality, rotation, translation)):
            raise ValueError('Valid model inputs must be finite')
        batch, length, count, _ = tokens.shape
        observed = mask.any(-1)
        encoded = self.token_norm(tokens)+self.position(xyz/3.)
        if state is None:
            memory = tokens.new_zeros(batch, self.num_joints, self.hidden_dim)
            context = features.new_zeros(batch, self.feature_dim)
            ready = torch.zeros(batch, dtype=torch.bool, device=tokens.device)
        else:
            memory, context, ready = state
        outputs = []
        for frame in range(length):
            available = observed[:, frame]
            measurement = features[:, frame]
            if self.reliability_gating:
                # Gate the direct observation path as well as memory; cold starts need a measurement.
                gain = torch.sigmoid(self.context_gate(torch.cat((measurement-context, quality[:, frame]), -1)))
                gain = torch.where(ready[:, None], gain, torch.ones_like(gain))
                measurement = context+gain*(measurement-context)
            context = torch.where(available[:, None], measurement, context)
            query = (self.joint_queries[None]+self.context_query(context)[:, None]
                     + self.memory_query(memory))
            valid = mask[:, frame].clone()
            valid[~available, 0] = True  # finite attention for empty frames; state is then held.
            points = encoded[:, frame].masked_fill(~mask[:, frame, :, None], 0.)
            joints, _ = self.attention(query, points, points, key_padding_mask=~valid, need_weights=False)
            q = self.quality(quality[:, frame])[:, None].expand(-1, self.num_joints, -1)
            candidate = self.memory(torch.cat((joints, q), -1).flatten(0, 1), memory.flatten(0, 1)).reshape_as(memory)
            if self.reliability_gating:
                gain = torch.sigmoid(self.joint_gate(torch.cat((joints, memory, q), -1)))
                gain = torch.where(ready[:, None, None], gain, torch.ones_like(gain))
                candidate = memory+gain*(candidate-memory)
            memory = torch.where(available[:, None, None], candidate, memory)
            ready = ready | available
            joint_features = context[:, None]+self.feature_update(memory)
            hidden = F.relu(self.head_hidden(self.head_norm(joint_features)))
            high_pose = torch.einsum('bjh,jch->bjc', hidden, self.joint_weight)+self.joint_bias
            low_pose = high_pose @ rotation[:, frame].transpose(-1, -2)+translation[:, frame, None]
            outputs.append(low_pose.masked_fill(~ready[:, None, None], 0.))
        return torch.stack(outputs, 1), (memory, context, ready)


def reset_state(state, reset):
    if state is None:
        return None
    return tuple(v.masked_fill(reset.reshape(-1, *([1]*(v.ndim-1))), 0) for v in state)


def detach_state(state):
    return tuple(v.detach() for v in state)
