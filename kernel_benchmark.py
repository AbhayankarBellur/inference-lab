
import statistics
import time

import torch
import triton
import triton.language as tl

assert torch.cuda.is_available()

DEVICE = "cuda"
REPEATS = 30
WARMUP = 10
SCALE = 1.25
BIAS = 0.5


@triton.jit
def fused_scale_bias(
    X, Y,
    N: tl.constexpr,
    SCALE: tl.constexpr,
    BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(X + offsets, mask=mask, other=0.0)
    y = x * SCALE + BIAS
    tl.store(Y + offsets, y, mask=mask)


def torch_reference(x):
    return x * SCALE + BIAS


def triton_operation(x, y):
    n = x.numel()
    block = 1024
    grid = (triton.cdiv(n, block),)

    fused_scale_bias[(grid[0],)](
        x, y, n,
        SCALE, BIAS,
        BLOCK=block,
    )
    return y


def benchmark(fn, x, y=None):
    # Compile / warm up outside the timed section.
    for _ in range(WARMUP):
        fn(x) if y is None else fn(x, y)

    torch.cuda.synchronize()
    timings_ms = []

    for _ in range(REPEATS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        fn(x) if y is None else fn(x, y)
        end.record()

        end.synchronize()
        timings_ms.append(start.elapsed_time(end))

    return statistics.median(timings_ms)


print("GPU:", torch.cuda.get_device_name(0))
print("Triton:", triton.__version__)

for n in [1024, 65536, 1_000_000, 4_000_000]:
    x = torch.randn(n, device=DEVICE, dtype=torch.float32)
    y = torch.empty_like(x)

    # Correctness check
    expected = torch_reference(x)
    actual = triton_operation(x, y)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected)

    # Compare two separate PyTorch operations with one Triton kernel.
    torch_ms = benchmark(torch_reference, x)
    triton_ms = benchmark(triton_operation, x, y)

    speedup = torch_ms / triton_ms if triton_ms > 0 else float("inf")

    print(f"\nElements: {n:,}")
    print(f"PyTorch (two operations): {torch_ms:.6f} ms")
    print(f"Triton (fused):           {triton_ms:.6f} ms")
    print(f"Speedup:                  {speedup:.2f}x")
    print("Correctness:              PASS")
