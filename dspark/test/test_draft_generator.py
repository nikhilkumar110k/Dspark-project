
from draft_generator import DraftGenerator


class ToyDraftModel:
    def next_token_logits(self, context):
        return [2.0, 1.0, 0.0]


def test_draft_generation():

    model = ToyDraftModel()

    generator = DraftGenerator(
        model=model,
        temperature=1.0,
        seed=42
    )

    prompt = [10, 20]

    result = generator.generate(
        prompt_tokens=prompt,
        max_tokens=3
    )

    print("Generated tokens:", result.tokens)

    print("Probability distributions:")

    for i, distribution in enumerate(result.probablities):
        print(f"Position {i + 1}: {distribution}")

    print("Sampled token probabilities:", result.sampled_probablities)

    assert len(result.tokens) == 3

    assert len(result.probablities) == 3

    for distribution in result.probablities:
        assert abs(sum(distribution) - 1.0) < 1e-9

    for token in result.tokens:
        assert 0 <= token < 3

    for token, distribution, sampled_probability in zip(
        result.tokens,
        result.probablities,
        result.sampled_probablities
    ):
        assert abs(
            distribution[token] - sampled_probability
        ) < 1e-9

    print("Draft generation test passed!")


if __name__ == "__main__":
    test_draft_generation()
