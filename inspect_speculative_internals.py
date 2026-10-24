
import inspect
import transformers
from transformers.generation.utils import GenerationMixin

print("Transformers:", transformers.__version__)

# Find the actual assisted-decoding implementation in this installation.
for name in dir(GenerationMixin):
    if "assisted" in name.lower():
        obj = getattr(GenerationMixin, name)
        if callable(obj):
            print(f"\n--- GenerationMixin.{name} ---")
            try:
                print("Signature:", inspect.signature(obj))
                source = inspect.getsource(obj)
                for i, line in enumerate(source.splitlines(), 1):
                    if any(term in line.lower() for term in (
                        "num_matches",
                        "update_candidate_strategy",
                        "candidate_generator",
                        "assistant_tokens",
                    )):
                        print(f"{i:4}: {line.strip()}")
            except (OSError, TypeError) as exc:
                print("Source unavailable:", exc)

# Inspect the candidate-strategy update method.
try:
    from transformers.generation.candidate_generator import (
        AssistedCandidateGenerator,
    )

    print("\n--- AssistedCandidateGenerator.update_candidate_strategy ---")
    method = AssistedCandidateGenerator.update_candidate_strategy
    print("Signature:", inspect.signature(method))
    print(inspect.getsource(method))
except (ImportError, AttributeError) as exc:
    print("\nCandidate generator location differs:", exc)
