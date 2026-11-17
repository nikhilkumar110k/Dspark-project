
import math

from verifier import Verifier


def test_softmax():

    logits = [2.0, 1.0, 0.0]

    probabilities = Verifier.softmax(logits)

    expected = [
        0.6652409558,
        0.2447284711,
        0.0900305732
    ]

    for actual, target in zip(probabilities, expected):
        assert abs(actual - target) < 1e-8

    assert abs(sum(probabilities) - 1.0) < 1e-9

    print("Softmax test passed!")


def test_all_tokens_accepted():

    verifier = Verifier(seed=42)

    Q = [
        [0.5, 0.3, 0.2],
        [0.5, 0.3, 0.2]
    ]

    target_logits = [
        [math.log(0.5), math.log(0.3), math.log(0.2)],
        [math.log(0.5), math.log(0.3), math.log(0.2)],

        [math.log(0.5), math.log(0.3), math.log(0.2)]
    ]

    draft_tokens = [0, 1]

    result = verifier.verify(
        draft_tokens=draft_tokens,
        draft_probablities=Q,
        target_logits=target_logits
    )

    assert result.all_accepted
    assert result.accepted_tokens == (0, 1)
    assert result.accepted_length == 2
    assert result.replacement_token is None
    assert result.bonus_token is not None

    print("All-token acceptance test passed!")
    print("Accepted tokens:", result.accepted_tokens)
    print("Bonus token:", result.bonus_token)


if __name__ == "__main__":
    test_softmax()
    test_all_tokens_accepted()

    print("\nAll verifier tests passed!")
