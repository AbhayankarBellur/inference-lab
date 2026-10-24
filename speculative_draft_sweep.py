
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
DRAFT_LENGTHS = [3, 5, 8]

PROMPT = (
    "Explain autoregressive language model inference in detail. "
    "Discuss token generation, attention, key-value caching, GPU memory "
    "bandwidth, decoding latency, and throughput in model serving. "
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


def run_trial(target, draft, input_ids, tokenizer, draft_length):
    config = GenerationConfig(
        max_new_tokens=OUTPUT_TOKENS,
        min_new_tokens=OUTPUT_TOKENS,
        do_sample=False,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        num_assistant_tokens=draft_length,
        num_assistant_tokens_schedule="constant",
    )

    def generate():
        return target.generate(
            input_ids=input_ids,
            generation_config=config,
            assistant_model=draft,
        )

    # Warmup outside the measured interval.
    with torch.inference_mode():
        generate()

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
            result = generate()
        synchronize()

        elapsed = time.perf_counter() - start
        token_ids = result[0, input_ids.shape[1]:].detach().cpu().tolist()
        peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)

        if len(token_ids) != OUTPUT_TOKENS:
            raise RuntimeError(
                f"Expected {OUTPUT_TOKENS} tokens; got {len(token_ids)}."
            )

        times.append(elapsed)
        throughputs.append(len(token_ids) / elapsed)
        peaks.append(peak_mib)
        outputs.append(token_ids)

        print(
            f"  Trial {trial + 1}: {elapsed:.3f}s | "
            f"{len(token_ids)} tokens | "
            f"{len(token_ids) / elapsed:.2f} tok/s | "
            f"{peak_mib:.1f} MiB"
        )

    return {
        "draft_length": draft_length,
        "elapsed": statistics.median(times),
        "throughput": statistics.median(throughputs),
        "peak_mib": statistics.median(peaks),
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
            "Target and draft tokenizers do not have identical vocabularies."
        )

    input_ids = tokenizer(
        PROMPT,
        return_tensors="pt",
    )["input_ids"].to(DEVICE)

    print("Prompt tokens:", input_ids.shape[1])
    print("Output tokens:", OUTPUT_TOKENS)

    target = load_model(TARGET_NAME)
    draft = None
    rows = []

    try:
        draft = load_model(DRAFT_NAME)

        for length in DRAFT_LENGTHS:
            print(f"\n=== Draft length: {length} ===")
            row = run_trial(
                target, draft, input_ids, tokenizer, length
            )
            rows.append(row)

    finally:
        if draft is not None:
            del draft
        del target
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 76)
    print("SPECULATIVE DRAFT-LENGTH SWEEP")
    print("=" * 76)
    print(
        f"{'Draft length':>13} {'Median s':>12} "
        f"{'Tokens/sec':>12} {'Peak MiB':>12} {'Output match':>15}"
    )
    print("-" * 76)

    reference = rows[0]["output"]

    for row in rows:
        print(
            f"{row['draft_length']:>13} "
            f"{row['elapsed']:>12.3f} "
            f"{row['throughput']:>12.2f} "
            f"{row['peak_mib']:>12.1f} "
            f"{str(row['output'] == reference):>15}"
        )

    print(
        "\nOutput equality is checked against the first assisted run. "
        "It is not an acceptance-rate measurement."
    )


if __name__ == "__main__":
    main()
