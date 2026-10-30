import time

import torch


def _count_sparse_conv(module, inputs, output):
    """Logical active-pair MACs; excludes padded work inside GPU kernels."""
    if module.conv1x1:
        pairs = output.features.shape[0]
    else:
        data = output.indice_dict[module.indice_key]
        if not hasattr(data, 'pair_fwd'):
            raise NotImplementedError('Sparse MAC counting requires spconv implicit-GEMM indices')
        indices = data.pair_bwd if module.inverse else data.pair_fwd
        pairs = int((indices >= 0).sum().item())
    module.total_ops += pairs * module.in_channels * module.out_channels


def _count_attention(module, inputs, output):
    query, key, value = inputs[:3]
    if query.ndim == 2:
        batch, nq, nk, nv = 1, query.shape[0], key.shape[0], value.shape[0]
    else:
        bdim, tdim = (0, 1) if module.batch_first else (1, 0)
        batch, nq, nk, nv = query.shape[bdim], query.shape[tdim], key.shape[tdim], value.shape[tdim]
    dim = module.embed_dim
    projections = batch * dim * (nq * query.shape[-1] + nk * key.shape[-1]
                                + nv * value.shape[-1] + nq * dim)
    extra = int(module.bias_k is not None) + int(module.add_zero_attn)
    module.total_ops += projections + 2 * batch * nq * (nk + extra) * dim


def _profile_macs(model, inputs):
    from thop import profile

    modules = list(model.modules())
    custom_ops = {torch.nn.MultiheadAttention: _count_attention}
    sparse = [m for m in modules if type(m).__module__ == 'spconv.pytorch.conv'
              and hasattr(m, 'kernel_size')]
    # Profiling needs pair maps even for convolutions that normally do not cache them.
    original_keys = [(m, m.indice_key) for m in sparse]
    state = [(m, m.training, set(m._forward_hooks),
              {k: m._buffers[k] for k in ('total_ops', 'total_params') if k in m._buffers})
             for m in modules]
    try:
        for index, module in enumerate(sparse):
            custom_ops[type(module)] = _count_sparse_conv
            if module.indice_key is None:
                module.indice_key = f'__profile_{id(module)}_{index}'
        # Fused Transformer paths can bypass child modules and their counting hooks.
        fastpath = torch.backends.mha.get_fastpath_enabled()
        try:
            torch.backends.mha.set_fastpath_enabled(False)
            macs, _ = profile(model, inputs=inputs, custom_ops=custom_ops, verbose=False)
        finally:
            torch.backends.mha.set_fastpath_enabled(fastpath)
        return macs
    finally:
        for module, key in original_keys:
            module.indice_key = key
        # THOP leaves buffers on unregistered modules, and hooks behind if forward fails.
        for module, training, hooks, buffers in state:
            module.training = training
            for key in set(module._forward_hooks) - hooks:
                del module._forward_hooks[key]
            for key in ('total_ops', 'total_params'):
                module._buffers.pop(key, None)
            module._buffers.update(buffers)


def _output_shape(output):
    if isinstance(output, torch.Tensor):
        return tuple(output.shape)
    if isinstance(output, dict):
        return {key: _output_shape(value) for key, value in output.items()}
    if isinstance(output, (list, tuple)):
        return [_output_shape(value) for value in output]
    return type(output).__name__


def _input_device(x):
    if isinstance(x, torch.Tensor):
        return x.device
    if isinstance(x, dict):
        for value in x.values():
            return _input_device(value)
    if isinstance(x, (list, tuple)):
        for value in x:
            return _input_device(value)
    raise TypeError(f"Cannot resolve device from input type: {type(x)}")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _forward(model, model_inputs):
    """Run a model with either one input or multiple positional inputs."""
    if isinstance(model_inputs, tuple):
        return model(*model_inputs)
    return model(model_inputs)


def _time_forward(model, x, device, warmup=20, repeat=100):
    with torch.inference_mode():
        for _ in range(warmup):
            _forward(model, x)
        _sync(device)

        if device.type == "cuda":
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            for _ in range(repeat):
                _forward(model, x)
            ender.record()
            _sync(device)
            return starter.elapsed_time(ender) / repeat

        start = time.perf_counter()
        for _ in range(repeat):
            _forward(model, x)
        return (time.perf_counter() - start) * 1000.0 / repeat


def profile_model(model_name, model, x, warmup=20, repeat=100):
    training = [(module, module.training) for module in model.modules()]
    try:
        return _profile_model(model_name, model, x, warmup, repeat)
    finally:
        for module, flag in training:
            module.training = flag


def _profile_model(model_name, model, x, warmup, repeat):
    if warmup < 0 or repeat < 1:
        raise ValueError('warmup must be nonnegative and repeat must be positive')
    device = _input_device(x)
    model.eval()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        output = _forward(model, x)
    _sync(device)

    infer_ms = _time_forward(model, x, device, warmup=warmup, repeat=repeat)

    peak_allocated_mb = None
    peak_reserved_mb = None
    if device.type == "cuda":
        peak_allocated_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / 1024 ** 2

    params_m = sum(p.numel() for p in model.parameters()) / 1000 ** 2
    try:
        thop_inputs = x if isinstance(x, tuple) else (x,)
        macs = _profile_macs(model, thop_inputs)
        macs_m = macs / 1000 ** 2
        flops_m = 2 * macs_m
    except Exception as exc:
        macs_m = None
        flops_m = None
        print(f"THOP_ERROR = {type(exc).__name__}: {exc}")

    print("===== Model Profile =====")
    print(f"Model = {model_name}")
    print(f"Device = {device}")
    print(f"Input shape = {_output_shape(x)}")
    print(f"Output shape = {_output_shape(output)}")
    print(f"FLOPs = {'N/A' if flops_m is None else f'{flops_m:.6f}'} M")
    print(f"MACs = {'N/A' if macs_m is None else f'{macs_m:.6f}'} M")
    print(f"Params = {params_m:.6f} M")
    print('MAC/FLOP scope = whole input forward; registered-module estimate, FLOPs = 2 * MACs')
    print('Includes sparse active-pair convolutions and MHA projections/QK/AV; excludes unregistered modules/functional ops and kernel padding.')
    print(f"Inference time = {infer_ms:.6f} ms/batch (whole input, not per frame)")
    if peak_allocated_mb is None:
        print("Memory allocated = N/A MB")
        print("Memory reserved = N/A MB")
    else:
        print(f"Memory allocated = {peak_allocated_mb:.6f} MB")
        print(f"Memory reserved = {peak_reserved_mb:.6f} MB")
