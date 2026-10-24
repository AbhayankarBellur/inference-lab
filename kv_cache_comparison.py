
import gc
import statistics
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
DEVICE = "cuda:0"
OUTPUT_TOKENS = 48
REPETITIONS = 3

PROMPT = (
    "Explain how transformer inference works, including tokenization, "
    "the attention mechanism, key-value caching, memory bandwidth, "
    "autoregressive decoding, and why inference servers use batching. "
) * 5


def synchronize():
    torch.cuda.synchronize()


def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        attn_implementation="sdpa",
    )
    model.to(DEVICE)
    model.eval()
    return model


def benchmark(model, tokenizer, input_ids, use_cache):
    config = GenerationConfig(
        max_new_tokens=OUTPUT_TOKENS,
        min_new_tokens=OUTPUT_TOKENS,
        do_sample=False,
        use_cache=use_cache,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Warmup is excluded from measurements.
    with torch.inference_mode():
        model.generate(input_ids=input_ids, generation_config=config)

    synchronize()
    times = []
    peaks = []
    token_counts = []
    last_output = None

    for trial in range(REPETITIONS):
        gc.collect()
        synchronize()
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                generation_config=config,
            )
        synchronize()

        elapsed = time.perf_counter() - start
        generated = output[0, input_ids.shape[1]:]
        count = generated.numel()
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

        times.append(elapsed)
        peaks.append(peak)
        token_counts.append(count)
        last_output = generated.detach().cpu().tolist()

        print(
            f"Trial {trial + 1}: {elapsed:.3f}s | "
            f"{count} tokens | {count / elapsed:.2f} tok/s | "
            f"{peak:.1f} MiB"
        )

    return {
        "use_cache": use_cache,
        "elapsed": statistics.median(times),
        "tokens": statistics.median(token_counts),
        "throughput": statistics.median(
            [n / t for n, t in zip(token_counts, times)]
        ),
        "peak": statistics.median(peaks),
        "output": last_output,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    print("Transformers:", transformers.__version__)
    print("PyTorch:", torch.__version__)
    print("GPU:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    input_ids = tokenizer(
        PROMPT,
        return_tensors="pt",
    )["input_ids"].to(DEVICE)

    print("Prompt tokens:", input_ids.shape[1])
    print("Generated tokens per run:", OUTPUT_TOKENS)

    model = load_model()

    try:
        print("\n=== KV CACHE ENABLED ===")
        cached = benchmark(model, tokenizer, input_ids, True)

        print("\n=== KV CACHE DISABLED ===")
        uncached = benchmark(model, tokenizer, input_ids, False)

        print("\n" + "=" * 68)
        print("KV CACHE COMPARISON")
        print("=" * 68)
        print(
            f"{'Metric':<28}{'Cache ON':>18}{'Cache OFF':>18}"
        )
        print("-" * 64)
        print(
            f"{'Median elapsed (s)':<28}"
            f"{cached['elapsed']:>18.3f}"
            f"{uncached['elapsed']:>18.3f}"
        )
        print(
            f"{'Tokens/sec (median)':<28}"
            f"{cached['throughput']:>18.2f}"
            f"{uncached['throughput']:>18.2f}"
        )
        print(
            f"{'Peak VRAM (MiB)':<28}"
            f"{cached['peak']:>18.1f}"
            f"{uncached['peak']:>18.1f}"
        )

        if cached["elapsed"] > 0:
            print(
                "\nElapsed-time ratio (OFF / ON): "
                f"{uncached['elapsed'] / cached['elapsed']:.2f}x"
            )

        print(
            "\nOutput token IDs identical:",
            cached["output"] == uncached["output"],
        )

    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
