"""Export the saved P4Transformer using standard ONNX operators (CPU supported)."""

import argparse
import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml

from models.P4Transformer import pointnet2_utils


def gather(features, indices):
    flat = indices.long().reshape(indices.shape[0], -1)
    result = features.gather(2, flat[:, None].expand(-1, features.shape[1], -1))
    return result.reshape(features.shape[0], features.shape[1], *indices.shape[1:])


def fps(xyz, count):
    count = int(count)
    size = int(xyz.shape[1])
    threads = 2 ** (size.bit_length() - 1)
    bits = threads.bit_length() - 1
    # Match CUDA's reduction tie order, including each thread's strided scan.
    order = [int(f'{i:0{bits}b}'[::-1], 2) for i in range(threads)]
    order = [j for i in order for j in range(i, size, threads)]
    order = torch.tensor(order, device=xyz.device)
    distances = torch.full_like(xyz[..., 0], 1e10)
    valid = xyz.square().sum(-1) > 1e-3
    current = torch.zeros(xyz.shape[0], 1, dtype=torch.long, device=xyz.device)
    selected = [current]
    for _ in range(1, count):
        anchor = xyz.gather(1, current[..., None].expand(-1, -1, 3))
        distances = torch.minimum(distances, (xyz - anchor).square().sum(-1))
        ranked = torch.where(valid, distances, -torch.ones_like(distances))[:, order]
        current = order[ranked.argmax(-1)].unsqueeze(-1)
        selected.append(current)
    return torch.cat(selected, dim=1)


def ball_query(radius, count, xyz, anchors):
    size = int(xyz.shape[1])
    distances = (anchors[:, :, None] - xyz[:, None]).square().sum(-1)
    indices = torch.arange(size, device=xyz.device).expand_as(distances)
    indices = torch.where(distances < radius * radius, indices, size)
    indices = indices.topk(count, dim=-1, largest=False, sorted=True).values
    first = torch.where(indices[..., :1] < size, indices[..., :1], 0)
    return torch.where(indices < size, indices, first)


class ExportModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, points, mask):
        return self.model({'input': points, 'mask': mask})['pose']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', type=Path)
    args = parser.parse_args()
    config_dir = args.experiment / 'config'
    config = yaml.safe_load((config_dir / 'config.yaml').read_text())
    model_config = yaml.safe_load((config_dir / 'model_config.yaml').read_text())
    spec = importlib.util.spec_from_file_location('saved_p4', config_dir / 'P4Transformer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.P4Transformer(**model_config)
    checkpoint = torch.load(args.experiment / 'checkpoint/last.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    del checkpoint
    model = ExportModel(model).eval()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    shape = (1, config['data']['T'], config['data']['person_max_points'])
    points = torch.randn(*shape, 3 + model_config['feat_dim'])
    points[..., 0] += 2
    mask = torch.rand(*shape) > 0.2
    mask[:, 0] = False
    points[~mask] = 0
    output = args.experiment / 'checkpoint/last_dynamic_batch.onnx'
    assert not output.exists(), f'Refusing to overwrite {output}'
    with patch.multiple(pointnet2_utils, furthest_point_sample=fps,
                        gather_operation=gather, grouping_operation=gather,
                        ball_query=ball_query), torch.inference_mode():
        # Empty neighborhoods, radius boundary, and repeated neighbor padding.
        xyz = torch.tensor([[[1., 0., 0.], [1.05, 0., 0.], [2., 0., 0.]]])
        assert ball_query(0.1, 3, xyz, xyz[:, :1]).tolist() == [[[0, 1, 0]]]
        assert ball_query(0.1, 3, xyz, xyz[:, :1] + 10).tolist() == [[[0, 0, 0]]]
        assert fps(torch.zeros(1, 4, 3), 2).tolist() == [[0, 0]]
        torch.onnx.export(model, (points, mask), str(output), opset_version=17,
                          dynamo=False, input_names=['points', 'mask'],
                          output_names=['pose'],
                          dynamic_axes={name: {0: 'batch'} for name in ('points', 'mask', 'pose')})
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(str(output))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(output), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    with patch.multiple(pointnet2_utils, furthest_point_sample=fps,
                        gather_operation=gather, grouping_operation=gather,
                        ball_query=ball_query), torch.inference_mode():
        for batch in (1, 3):
            points = torch.randn(batch, *shape[1:], 3 + model_config['feat_dim'])
            points[..., 0] += 2
            mask = torch.rand(batch, *shape[1:]) > 0.2
            mask[:, 0] = False
            points[~mask] = 0
            expected = model(points, mask).numpy()
            actual = session.run(None, {'points': points.numpy(), 'mask': mask.numpy()})[0]
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
            assert actual.shape == (batch, shape[1], 1, model_config['num_joints'], 3)
            assert np.isfinite(actual).all()
            for i in range(batch):
                single = session.run(None, {'points': points[i:i+1].numpy(),
                                            'mask': mask[i:i+1].numpy()})[0]
                np.testing.assert_allclose(actual[i:i+1], single, rtol=1e-4, atol=1e-4)
            print(f'Batch {batch}: output {actual.shape}; max absolute error: '
                  f'{np.max(np.abs(actual - expected)):.8g}')
    print(f'Exported: {output}')


if __name__ == '__main__':
    main()
