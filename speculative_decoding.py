
import gc
import statistics
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

TARGET_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
DRAFT_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"

DEVICE = "cuda:0"
OUTPUT_TOKENS = 96
REPETITIONS = 3
DRAFT_TOKENS = 5

PROMPT = (
    "Explain autoregressive language model inference in detail. "
    "Discuss token generation, attention, key-value caching, GPU memory "
    "bandwidth, decoding latency, and the trade-offs between throughput "
    "and latency in a model-serving system. "
) * 4


def synchronize():
    torch.cuda.synchronize()


def load_model(name):
    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=torch.float16,
        attn_implementation="sdpa",
    )
    model.to(DEVICE)
    model.eval()
    return model


def run_benchmark(target, draft, input_ids, tokenizer, assisted):
    config = GenerationConfig(
        max_new_tokens=OUTPUT_TOKENS,
        min_new_tokens=OUTPUT_TOKENS,
        do_sample=False,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        num_assistant_tokens=DRAFT_TOKENS,
    )

    def generate():
        kwargs = {
            "input_ids": input_ids,
            "generation_config": config,
        }
        if assisted:
            kwargs["assistant_model"] = draft
        return target.generate(**kwargs)

    # Warmup is excluded from measurement.
    with torch.inference_mode():
        generate()

    synchronize()
    times = []
    peaks = []
    outputs = []

    for trial in range(REPETITIONS):
        gc.collect()
        synchronize()
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        with torch.inference_mode():
            result = generate()
        synchronize()

        elapsed = time.perf_counter() - start
        generated = result[0, input_ids.shape[1]:]
        count = generated.numel()
        peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)

        times.append(elapsed)
        peaks.append(peak_mib)
        outputs.append(generated.detach().cpu().tolist())

        print(
            f"Trial {trial + 1}: {elapsed:.3f}s | "
            f"{count} tokens | {count / elapsed:.2f} tok/s | "
            f"{peak_mib:.1f} MiB VRAM"
        )

    median_time = statistics.median(times)
    median_throughput = statistics.median(
        [OUTPUT_TOKENS / t for t in times]
    )

    return {
        "elapsed": median_time,
        "throughput": median_throughput,
        "peak": statistics.median(peaks),
        "output": outputs[-1],
        "all_outputs": outputs,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    print("Transformers:", transformers.__version__)
    print("PyTorch:", torch.__version__)
    print("GPU:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(TARGET_NAME)
    draft_tokenizer = AutoTokenizer.from_pretrained(DRAFT_NAME)

    if tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise RuntimeError(
            "The target and draft tokenizers do not have identical "
            "vocabularies. Do not compare these models with this script."
        )

    input_ids = tokenizer(
        PROMPT,
        return_tensors="pt",
    )["input_ids"].to(DEVICE)

    print("Prompt tokens:", input_ids.shape[1])
    print("Required output tokens:", OUTPUT_TOKENS)
    print("Draft tokens per proposal:", DRAFT_TOKENS)

    target = load_model(TARGET_NAME)
    draft = None

    try:
        print("\n=== Target model: ordinary greedy decoding ===")
        baseline = run_benchmark(
            target, None, input_ids, tokenizer, assisted=False
        )

        print("\nLoading draft model...")
        draft = load_model(DRAFT_NAME)

        print("\n=== Assisted generation: speculative decoding ===")
        speculative = run_benchmark(
            target, draft, input_ids, tokenizer, assisted=True
        )

        print("\n" + "=" * 68)
        print("SPECULATIVE DECODING COMPARISON")
        print("=" * 68)
        print(
            f"{'Metric':<28}{'Ordinary':>18}{'Speculative':>18}"
        )
        print("-" * 64)
        print(
            f"{'Median elapsed (s)':<28}"
            f"{baseline['elapsed']:>18.3f}"
            f"{speculative['elapsed']:>18.3f}"
        )
        print(
            f"{'Throughput (tok/s)':<28}"
            f"{baseline['throughput']:>18.2f}"
            f"{speculative['throughput']:>18.2f}"
        )
        print(
            f"{'Peak VRAM (MiB)':<28}"
            f"{baseline['peak']:>18.1f}"
            f"{speculative['peak']:>18.1f}"
        )

        if speculative["elapsed"] > 0:
            print(
                "\nSpeedup ratio: "
                f"{baseline['elapsed'] / speculative['elapsed']:.2f}x"
            )

        same = (
            baseline["output"] == speculative["output"]
        )
        print("Final output token IDs identical:", same)

        if not same:
            print(
                "WARNING: outputs differ. Inspect generation settings, "
                "tokenizer compatibility, and stopping behavior."
            )

        print("\nOrdinary output:")
        print(tokenizer.decode(
            baseline["output"], skip_special_tokens=True
        ))
        print("\nSpeculative output:")
        print(tokenizer.decode(
            speculative["output"], skip_special_tokens=True
        ))

    finally:
        if draft is not None:
            del draft
        del target
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
