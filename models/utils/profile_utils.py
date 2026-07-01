import os
import time

import torch


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


def _time_forward(model, x, device, warmup=20, repeat=100):
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        _sync(device)

        if device.type == "cuda":
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            for _ in range(repeat):
                model(x)
            ender.record()
            _sync(device)
            return starter.elapsed_time(ender) / repeat

        start = time.perf_counter()
        for _ in range(repeat):
            model(x)
        return (time.perf_counter() - start) * 1000.0 / repeat


def profile_model(model_name, model, x, warmup=20, repeat=100):
    device = _input_device(x)
    model.eval()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        output = model(x)
    _sync(device)

    infer_ms = _time_forward(model, x, device, warmup=warmup, repeat=repeat)

    peak_allocated_mb = None
    peak_reserved_mb = None
    if device.type == "cuda":
        peak_allocated_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / 1024 ** 2

    try:
        from thop import profile

        flops, params = profile(model, inputs=(x,), verbose=False)
        flops_m = flops / 1000 ** 2
        params_m = params / 1000 ** 2
    except Exception as exc:
        flops_m = None
        params_m = sum(p.numel() for p in model.parameters()) / 1000 ** 2
        print(f"THOP_ERROR = {type(exc).__name__}: {exc}")

    print("===== Model Profile =====")
    print(f"Model = {model_name}")
    print(f"Device = {device}")
    print(f"Input shape = {_output_shape(x)}")
    print(f"Output shape = {_output_shape(output)}")
    print(f"FLOPs = {'N/A' if flops_m is None else f'{flops_m:.6f}'} M")
    print(f"Params = {params_m:.6f} M")
    print(f"Inference time = {infer_ms:.6f} ms/sample")
    if peak_allocated_mb is None:
        print("Memory allocated = N/A MB")
        print("Memory reserved = N/A MB")
    else:
        print(f"Memory allocated = {peak_allocated_mb:.6f} MB")
        print(f"Memory reserved = {peak_reserved_mb:.6f} MB")
