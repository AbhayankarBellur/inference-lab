
import gc
import csv
import time
import statistics
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
PROMPT = "Explain what GPU inference is in simple terms."
OUTPUT_TOKENS = 64
WARMUP_TOKENS = 8
REPETITIONS = 3

assert torch.cuda.is_available(), "CUDA is unavailable"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
messages = [{"role": "user", "content": PROMPT}]
formatted = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(formatted, return_tensors="pt").to(device)
input_tokens = inputs["input_ids"].shape[1]


def load_model(mode):
    if mode == "fp16":
        return AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        ).to(device).eval()

    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=config,
        device_map={"": 0},
        attn_implementation="sdpa",
    ).eval()


def run_once(model):
    # Prefill and first-token selection
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        out = model(**inputs, use_cache=True)
        token = out.logits[:, -1, :].argmax(
            dim=-1, keepdim=True
        )
        past = out.past_key_values

    torch.cuda.synchronize()
    ttft = time.perf_counter() - start

    # Decode the remaining 63 tokens using the KV cache
    with torch.inference_mode():
        start = time.perf_counter()

        for _ in range(OUTPUT_TOKENS - 1):
            out = model(
                input_ids=token,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            token = out.logits[:, -1, :].argmax(
                dim=-1, keepdim=True
            )

        torch.cuda.synchronize()
        decode_time = time.perf_counter() - start

    return ttft, decode_time


def benchmark(mode):
    print(f"\nLoading {mode.upper()} model...")
    model = load_model(mode)

    # Warm-up is excluded from measured results.
    run_once(model)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    ttfts = []
    decode_times = []

    for i in range(REPETITIONS):
        ttft, decode_time = run_once(model)
        ttfts.append(ttft)
        decode_times.append(decode_time)

        print(
            f"Run {i + 1}/{REPETITIONS}: "
            f"TTFT={ttft * 1000:.2f} ms, "
            f"decode={(OUTPUT_TOKENS - 1) / decode_time:.2f} tok/s"
        )

    peak_mib = torch.cuda.max_memory_allocated() / 1024**2
    median_ttft = statistics.median(ttfts)
    median_decode = statistics.median(decode_times)

    result = {
        "experiment": mode,
        "model": MODEL_ID,
        "input_tokens": input_tokens,
        "output_tokens": OUTPUT_TOKENS,
        "ttft_ms": round(median_ttft * 1000, 2),
        "decode_tps": round((OUTPUT_TOKENS - 1) / median_decode, 2),
        "total_latency_s": round(median_ttft + median_decode, 3),
        "peak_vram_mib": round(peak_mib, 2),
    }

    print(f"\n{mode.upper()} MEDIAN")
    for key, value in result.items():
        print(f"{key}: {value}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


results = []

for mode in ("fp16", "4bit_nf4"):
    results.append(benchmark(mode))

# Write a separate file so the original results remain untouched.
with Path("normalized_results.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

fp16, quantized = results
print("\n--- NORMALIZED COMPARISON ---")
print(
    "Decode throughput change: "
    f"{(quantized['decode_tps'] / fp16['decode_tps'] - 1) * 100:+.2f}%"
)
print(
    "Peak allocated VRAM change: "
    f"{(quantized['peak_vram_mib'] / fp16['peak_vram_mib'] - 1) * 100:+.2f}%"
)
print("\nSaved normalized_results.csv")
