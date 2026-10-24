
import gc
import statistics
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.candidate_generator import AssistedCandidateGenerator

TARGET_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
DRAFT_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
DRAFT_LENGTHS = [3, 5, 8]
REPETITIONS = 3
NEW_TOKENS = 96

print("Transformers:", transformers.__version__)
print("GPU:", torch.cuda.get_device_name(0))

tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
target = AutoModelForCausalLM.from_pretrained(
    TARGET_ID,
    torch_dtype=torch.float16,
    attn_implementation="sdpa",
).to("cuda").eval()

draft = AutoModelForCausalLM.from_pretrained(
    DRAFT_ID,
    torch_dtype=torch.float16,
    attn_implementation="sdpa",
).to("cuda").eval()

# Both models must use compatible token IDs for ordinary assisted decoding.
assert target.config.vocab_size == draft.config.vocab_size
assert tokenizer.get_vocab() == AutoTokenizer.from_pretrained(DRAFT_ID).get_vocab()

prompt = (
    "Explain how speculative decoding improves language model inference. "
    "Discuss draft models, target verification, accepted tokens, and the "
    "trade-off between model overhead and decoding speed."
)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

original_update = AssistedCandidateGenerator.update_candidate_strategy

def run_one(draft_length):
    stats = {"matches": 0, "proposed": 0, "iterations": 0}

    def instrumented_update(self, input_ids, scores, num_matches):
        # For standard assisted decoding, scores has one target score
        # per candidate position, including the extra verification position.
        candidate_count = max(0, len(scores[0]) - 1)
        stats["matches"] += int(num_matches)
        stats["proposed"] += candidate_count
        stats["iterations"] += 1
        return original_update(self, input_ids, scores, num_matches)

    AssistedCandidateGenerator.update_candidate_strategy = instrumented_update

    try:
        # Explicitly request a constant draft length.
        draft.generation_config.num_assistant_tokens = draft_length
        draft.generation_config.num_assistant_tokens_schedule = "constant"

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()

        with torch.inference_mode():
            output = target.generate(
                **inputs,
                assistant_model=draft,
                max_new_tokens=NEW_TOKENS,
                min_new_tokens=NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=False,
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        generated = output.shape[1] - inputs["input_ids"].shape[1]
        peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)

        return {
            "elapsed": elapsed,
            "tokens": generated,
            "tok_s": generated / elapsed,
            "peak_mib": peak_mib,
            **stats,
        }
    finally:
        AssistedCandidateGenerator.update_candidate_strategy = original_update

try:
    print("\nDraft | Median s | Tok/s | Matches | Proposed | Acceptance | Peak MiB")
    print("-" * 78)

    for length in DRAFT_LENGTHS:
        # Warm-up, excluded from the measurements.
        run_one(length)
        gc.collect()
        torch.cuda.empty_cache()

        trials = [run_one(length) for _ in range(REPETITIONS)]
        median_s = statistics.median(x["elapsed"] for x in trials)
        median_tps = statistics.median(x["tok_s"] for x in trials)
        peak = max(x["peak_mib"] for x in trials)
        matches = sum(x["matches"] for x in trials)
        proposed = sum(x["proposed"] for x in trials)
        acceptance = 100 * matches / proposed if proposed else 0

        print(
            f"{length:5} | {median_s:8.3f} | {median_tps:5.2f} | "
            f"{matches:7} | {proposed:8} | {acceptance:9.2f}% | "
            f"{peak:8.1f}"
        )
finally:
    AssistedCandidateGenerator.update_candidate_strategy = original_update
