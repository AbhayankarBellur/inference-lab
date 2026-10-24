
import gc
import statistics
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import (
    ContinuousBatchingConfig,
    GenerationConfig,
)

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
DEVICE = "cuda:0"
DTYPE = torch.float16
MAX_NEW_TOKENS = 48
CONCURRENCY_LEVELS = [1, 2, 4, 8]
REPETITIONS = 3

PROMPTS = [
    "Explain GPU inference in simple terms.",
    "What is a transformer KV cache?",
    "Explain why memory bandwidth matters for LLM inference.",
    "Compare static batching with continuous batching.",
]


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


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    print("Transformers:", transformers.__version__)
    print("PyTorch:", torch.__version__)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Concurrency levels:", CONCURRENCY_LEVELS)
    print("Measured repetitions:", REPETITIONS)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    generation_config = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    model = load_model()
    rows = []

    try:
        for concurrency in CONCURRENCY_LEVELS:
            print(f"\n{'=' * 55}")
            print(f"CONCURRENCY: {concurrency}")
            print("=" * 55)

            prompts = [
                PROMPTS[i % len(PROMPTS)]
                for i in range(concurrency)
            ]
            encoded = [
                tokenizer.encode(prompt)
                for prompt in prompts
            ]

            batching_config = ContinuousBatchingConfig(
                max_memory_percent=0.20,
                page_size=16,
                scheduler_type="fifo",
                use_cuda_graph=False,
                max_requests_per_batch=concurrency,
            )

            # Warmup is excluded from measurements.
            warmup = model.generate_batch(
                inputs=[tokenizer.encode("Warmup request.")],
                generation_config=generation_config,
                continuous_batching_config=batching_config,
            )

            if not warmup:
                raise RuntimeError("Warmup returned no results.")

            del warmup
            gc.collect()
            synchronize()

            trial_times = []
            trial_tokens = []
            trial_vram = []
            all_trials_valid = True

            for trial in range(REPETITIONS):
                gc.collect()
                synchronize()
                torch.cuda.reset_peak_memory_stats()

                start = time.perf_counter()

                # Deliberately no outer inference_mode wrapper:
                # generate_batch manages its execution context.
                results = model.generate_batch(
                    inputs=encoded,
                    generation_config=generation_config,
                    continuous_batching_config=batching_config,
                )

                synchronize()
                elapsed = time.perf_counter() - start
                peak_mib = (
                    torch.cuda.max_memory_allocated() / (1024 ** 2)
                )

                expected = [
                    f"req_{i}" for i in range(concurrency)
                ]
                missing = [
                    request_id for request_id in expected
                    if request_id not in results
                ]

                if missing:
                    print(
                        f"Trial {trial + 1}: FAILED; "
                        f"missing {missing}; returned "
                        f"{list(results.keys())}"
                    )
                    all_trials_valid = False
                    break

                counts = [
                    len(results[request_id].generated_tokens)
                    for request_id in expected
                ]
                total_tokens = sum(counts)

                if total_tokens == 0:
                    print(f"Trial {trial + 1}: FAILED; zero tokens")
                    all_trials_valid = False
                    break

                trial_times.append(elapsed)
                trial_tokens.append(total_tokens)
                trial_vram.append(peak_mib)

                print(
                    f"Trial {trial + 1}: "
                    f"{elapsed:.3f}s | "
                    f"{total_tokens} tokens | "
                    f"{total_tokens / elapsed:.2f} tok/s | "
                    f"{peak_mib:.1f} MiB VRAM"
                )

            if not all_trials_valid:
                rows.append({
                    "concurrency": concurrency,
                    "valid": False,
                })
                continue

            median_time = statistics.median(trial_times)
            median_tokens = statistics.median(trial_tokens)
            median_vram = statistics.median(trial_vram)
            throughput = median_tokens / median_time

            row = {
                "concurrency": concurrency,
                "valid": True,
                "elapsed": median_time,
                "tokens": median_tokens,
                "throughput": throughput,
                "vram": median_vram,
            }
            rows.append(row)

            print(
                f"MEDIAN: {median_time:.3f}s | "
                f"{median_tokens:.0f} tokens | "
                f"{throughput:.2f} tok/s | "
                f"{median_vram:.1f} MiB"
            )

    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("\n\n" + "=" * 78)
    print("CONCURRENCY SCALING SUMMARY")
    print("=" * 78)
    print(
        f"{'Requests':>9} {'Median s':>11} {'Tokens':>9} "
        f"{'Tokens/s':>11} {'VRAM MiB':>12} {'vs 1 request':>15}"
    )
    print("-" * 78)

    baseline = next(
        (
            row["throughput"] for row in rows
            if row.get("valid") and row["concurrency"] == 1
        ),
        None,
    )

    for row in rows:
        if not row.get("valid"):
            print(f"{row['concurrency']:>9} {'FAILED':>11}")
            continue

        ratio = (
            row["throughput"] / baseline
            if baseline else float("nan")
        )
        print(
            f"{row['concurrency']:>9} "
            f"{row['elapsed']:>11.3f} "
            f"{row['tokens']:>9.0f} "
            f"{row['throughput']:>11.2f} "
            f"{row['vram']:>12.1f} "
            f"{ratio:>14.2f}x"
        )

    print("\nInterpretation:")
    print("- Higher aggregate tokens/sec means greater batch throughput.")
    print("- Higher VRAM indicates the memory cost of concurrent requests.")
    print("- This measures batched completion, not live request arrivals.")
    print("- A failed run is not a performance measurement.")


if __name__ == "__main__":
    main()
