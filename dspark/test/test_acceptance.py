
from acceptance import Acceptance
from residual_sample import ResidualSampler


def test_residual_calculation():

    sampler = ResidualSampler()

    p = [0.70, 0.20, 0.10]

    q = [0.50, 0.40, 0.10]

    residual = sampler.calculate_residual(p, q)

    assert all(
    abs(actual - expected) < 1e-9
    for actual, expected in zip(
        residual,
        [0.20, 0.0, 0.0]
    )
)

    normalized = sampler.normalize(residual)

    assert all(
    abs(actual - expected) < 1e-9
    for actual, expected in zip(
        normalized,
        [1.0, 0.0, 0.0]
    )
)


def test_acceptance_probability():

    acceptance = Acceptance(seed=42)

    alpha = acceptance.acceptance_probablity(
        0.30,
        0.50
    )

    assert abs(alpha - 0.60) < 1e-9


def test_always_accept():

    acceptance = Acceptance(seed=42)

    p = [0.70, 0.20, 0.10]

    q = [0.70, 0.20, 0.10]

    result = acceptance.verify_draft(
        draft_tokens=[0, 1],
        draft_probablities=[q, q],
        target_probablities=[p, p]
    )

    assert result.all_accepted

    assert result.accepted_length == 2

    assert result.replacement_token is None


def test_residual_correction():

    acceptance = Acceptance(seed=42)

    p = [0.70, 0.20, 0.10]

    q = [0.50, 0.40, 0.10]

    result = acceptance.verify_draft(
        draft_tokens=[1],
        draft_probablities=[q],
        target_probablities=[p]
    )

 

    if not result.all_accepted:

        assert result.accepted_length == 0

        assert result.replacement_token == 0


if __name__ == "__main__":

    test_residual_calculation()

    test_acceptance_probability()

    test_always_accept()

    test_residual_correction()

    print("All tests passed!")
