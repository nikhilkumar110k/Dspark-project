
from collections import Counter
import random

from acceptance import Acceptance

P = [0.70, 0.20, 0.10]
Q = [0.50, 0.40, 0.10]

rng = random.Random(42)
acceptance = Acceptance(seed=42)

iterations = 100000
counts = Counter()

for _ in range(iterations):

    # Generate one draft token from Q
    u = rng.random()
    cumulative = 0.0

    draft_token = None

    for i, probability in enumerate(Q):
        cumulative += probability

        if u < cumulative:
            draft_token = i
            break

    # Verify the proposed token
    result = acceptance.verify_draft(
        draft_tokens=[draft_token],
        draft_probablities=[Q],
        target_probablities=[P]
    )

    if result.all_accepted:
        output_token = draft_token
    else:
        output_token = result.replacement_token

    counts[output_token] += 1

print("Expected distribution:", P)
print("Observed distribution:")

for token in range(len(P)):
    print(
        f"Token {token}: "
        f"{counts[token] / iterations:.4f}"
    )
