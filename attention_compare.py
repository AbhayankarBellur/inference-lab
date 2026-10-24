
import gc
import csv
import time
import statistics
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
OUTPUT_TOKENS = 32
REPETITIONS = 3
MAX_INPUT_TOKENS = 384

assert torch.cuda.is_available()
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Longer prompt to make attention/prefill differences easier to see.
paragraph = (
    "GPU inference runs a trained neural network to generate predictions. "
    "The model processes input tokens and uses matrix operations and attention "
    "to compute the next token. Quantization reduces weight precision and can "
    "reduce memory use. Attention kernels can improve memory traffic and "
    "computation efficiency. Serving systems also use batching and KV caches "
    "to handle multiple requests efficiently. "
)
messages = [{
    "role": "user",
    "content": paragraph * 12 + "Summarize the main ideas."
}]

formatted = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(
    formatted,
    return_tensors="pt",
    truncation=True,
    max_length=MAX_INPUT_TOKENS,
).to(device)

input_tokens = inputs["input_ids"].shape[1]
print("GPU:", torch.cuda.get_device_name(0))
print("Input tokens:", input_tokens)


def load_model(implementation):
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        attn_implementation=implementation,
    ).to(device).eval()


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

    # Decode using cached key/value states
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


def benchmark(implementation):
    print(f"\nLoading attention={implementation}")
    model = load_model(implementation)

    run_once(model)  # warm-up
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    ttfts = []
    decode_times = []

    for i in range(REPETITIONS):
        ttft, decode_time = run_once(model)
        ttfts.append(ttft)
        decode_times.append(decode_time)
        print(
            f"Run {i + 1}: TTFT={ttft * 1000:.2f} ms, "
            f"decode={(OUTPUT_TOKENS - 1) / decode_time:.2f} tok/s"
        )

    result = {
        "experiment": implementation,
        "model": MODEL_ID,
        "input_tokens": input_tokens,
        "output_tokens": OUTPUT_TOKENS,
        "ttft_ms": round(statistics.median(ttfts) * 1000, 2),
        "decode_tps": round(
            (OUTPUT_TOKENS - 1) / statistics.median(decode_times), 2
        ),
        "peak_vram_mib": round(
            torch.cuda.max_memory_allocated() / 1024**2, 2
        ),
    }

    print("\nMedian:", result)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


results = []
for implementation in ("eager", "sdpa"):
    results.append(benchmark(implementation))

with Path("attention_results.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\n--- ATTENTION COMPARISON ---")
eager, sdpa = results
print(
    "SDPA decode throughput change: "
    f"{(sdpa['decode_tps'] / eager['decode_tps'] - 1) * 100:+.2f}%"
)
print(
    "SDPA TTFT change: "
    f"{(sdpa['ttft_ms'] / eager['ttft_ms'] - 1) * 100:+.2f}%"
)
print("Saved attention_results.csv")
