
import gc
import statistics
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
DEVICE = "cuda:0"
DTYPE = torch.float16

CONTEXT_LENGTHS = [176, 512, 1024, 1536, 2048]
OUTPUT_TOKENS = 48
REPETITIONS = 3


def synchronize():
    torch.cuda.synchronize()


def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE,
        attn_implementation="sdpa",
    )
    model.to(DEVICE)
    model.eval()
    return model


def build_prompt_ids(tokenizer, target_length):
    # Repeating token IDs guarantees the requested input length.
    # This is a controlled compute/memory benchmark, not natural prose.
    seed_text = (
        "Explain transformer inference, attention, KV caching, "
        "GPU memory bandwidth, and autoregressive token generation. "
    )
    seed_ids = tokenizer.encode(seed_text, add_special_tokens=False)

    if not seed_ids:
        raise RuntimeError("Seed prompt produced no tokens.")

    repetitions = (target_length + len(seed_ids) - 1) // len(seed_ids)
    token_ids = (seed_ids * repetitions)[:target_length]

    return torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=DEVICE,
    )


def benchmark(model, input_ids, use_cache):
    config = GenerationConfig(
        max_new_tokens=OUTPUT_TOKENS,
        min_new_tokens=OUTPUT_TOKENS,
        do_sample=False,
        use_cache=use_cache,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=model.generation_config.pad_token_id,
    )

    # Warmup excluded from measurements.
    with torch.inference_mode():
        model.generate(
            input_ids=input_ids,
            generation_config=config,
        )

    synchronize()

    times = []
    throughputs = []
    peaks = []
    outputs = []

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
        token_ids = generated.detach().cpu().tolist()
        token_count = len(token_ids)
        peak_mib = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
        )

        if token_count != OUTPUT_TOKENS:
            raise RuntimeError(
                f"Expected {OUTPUT_TOKENS} tokens, got {token_count}."
            )

        times.append(elapsed)
        throughputs.append(token_count / elapsed)
        peaks.append(peak_mib)
        outputs.append(token_ids)

        print(
            f"  Trial {trial + 1}: {elapsed:.3f}s | "
            f"{token_count} tokens | "
            f"{token_count / elapsed:.2f} tok/s | "
            f"{peak_mib:.1f} MiB"
        )

    return {
        "elapsed": statistics.median(times),
        "throughput": statistics.median(throughputs),
        "peak_mib": statistics.median(peaks),
        "output": outputs[-1],
        "outputs": outputs,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    print("Transformers:", transformers.__version__)
    print("PyTorch:", torch.__version__)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Context lengths:", CONTEXT_LENGTHS)
    print("Output tokens:", OUTPUT_TOKENS)
    print("Repetitions:", REPETITIONS)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_model()
    results = []

    try:
        for context_length in CONTEXT_LENGTHS:
            print("\n" + "=" * 68)
            print(f"CONTEXT LENGTH: {context_length} tokens")
            print("=" * 68)

            input_ids = build_prompt_ids(tokenizer, context_length)
            actual_length = input_ids.shape[1]
            print("Actual prompt tokens:", actual_length)

            # Test cache ON first, then OFF, with the same input.
            for use_cache in (True, False):
                label = "ON" if use_cache else "OFF"
                print(f"\nKV cache {label}")

                try:
                    row = benchmark(model, input_ids, use_cache)
                    row["context_length"] = actual_length
                    row["use_cache"] = use_cache
                    row["valid"] = True
                    results.append(row)

                    print(
                        f"  MEDIAN: {row['elapsed']:.3f}s | "
                        f"{row['throughput']:.2f} tok/s | "
                        f"{row['peak_mib']:.1f} MiB"
                    )

                except torch.cuda.OutOfMemoryError:
                    print(
                        "  CUDA OUT OF MEMORY: skipping this cache mode "
                        "at this context length."
                    )
                    results.append({
                        "context_length": actual_length,
                        "use_cache": use_cache,
                        "valid": False,
                    })
                    gc.collect()
                    torch.cuda.empty_cache()

                finally:
                    gc.collect()
                    torch.cuda.empty_cache()

    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("\n\n" + "=" * 92)
    print("LONG-CONTEXT KV CACHE SCALING SUMMARY")
    print("=" * 92)
    print(
        f"{'Context':>8} {'Cache':>8} {'Median s':>11} "
        f"{'Tokens/s':>11} {'Peak MiB':>12} {'Output matches':>16}"
    )
    print("-" * 92)

    cache_outputs = {}

    for row in results:
        if not row.get("valid"):
            print(
                f"{row['context_length']:>8} "
                f"{'ON' if row['use_cache'] else 'OFF':>8} "
                f"{'OOM/FAIL':>11}"
            )
            continue

        key = row["context_length"]
        cache_outputs.setdefault(key, {})[row["use_cache"]] = row["output"]

        if True in cache_outputs[key] and False in cache_outputs[key]:
            matches = (
                cache_outputs[key][True] == cache_outputs[key][False]
            )
            match_text = "Yes" if matches else "No"
        else:
            match_text = "Pending"

        print(
            f"{row['context_length']:>8} "
            f"{'ON' if row['use_cache'] else 'OFF':>8} "
            f"{row['elapsed']:>11.3f} "
            f"{row['throughput']:>11.2f} "
            f"{row['peak_mib']:>12.1f} "
            f"{match_text:>16}"
        )

    print("\nInterpretation:")
    print("- KV caching avoids recomputing previous tokens during decoding.")
    print("- Longer prompts increase prefill work and cache storage.")
    print("- Cache OFF may be much slower at longer context lengths.")
    print("- OOM means the tested configuration exceeded available memory.")
    print("- Peak VRAM is PyTorch allocated memory, not total GPU usage.")


if __name__ == "__main__":
    main()
