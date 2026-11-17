import math
import random
from dataclasses import dataclass
from typing import Sequence
from acceptance import Acceptance
from residual_sample import ResidualSampler, validate_distribution

@dataclass(frozen=True)
class VerificationResult:
    accepted_tokens: tuple[int,...]
    accepted_length: int
    rejected_index: int
    replacement_token: int | None
    bonus_token: int | None

    @property
    def all_accepted(self)->bool:
        return self.rejected_index == -1

class Verifier:
    def __init__(self,acceptance=None,seed=None):
        self.acceptance=acceptance or Acceptance(seed=seed)
        self.rng= random.Random(seed)
        self.sampler= ResidualSampler(rng=self.rng)

    @staticmethod
    def softmax(logits: Sequence[float])->list[float]:
        if len(logits) ==0:
            raise ValueError("Logits can't be zero")
        for value in logits:
            if not math.isfinite(value):
                raise ValueError("logits or values inside it can't be infinite")
        max_logit=max(logits)
        exponentials=[math.exp(value-max_logit)for value in logits]
        total = sum(exponentials)
        if total == 0.0 or not math.isfinite(total):
            raise ValueError("the total can't be zero or not finite")
        probablities= [value/total for value in exponentials]
        return probablities
    
    @classmethod
    def logits_to_probablties(cls,target_logits:Sequence[Sequence[float]])->list[list[float]]:
        if len(target_logits) == 0:
            raise ValueError("the target logits can't be zero")
        return [
            cls.softmax(position_logits)
            for position_logits in target_logits
        ]
    def verify(self,draft_tokens: Sequence[int],draft_probablities: Sequence[Sequence[float]],target_logits:Sequence[Sequence[float]]) -> VerificationResult:
        draft_length = len(draft_tokens)
        if draft_length == 0:
            raise ValueError("the draft length can't be zero")
        if len(draft_probablities) != draft_length:
            raise ValueError("the length of draft probs and draft length don't match")
        if len(target_logits) != draft_length +1:
            raise ValueError("target must return k+1 postions: "
                             "K verification distributions and one bonus distribution")
        target_probablities= self.logits_to_probablties(target_logits)
        result= self.acceptance.verify_draft(
            draft_tokens=draft_tokens,
            draft_probablities=draft_probablities,
            target_probablities=target_probablities[:draft_length]
        )
        if result.all_accepted:
            bonus_token=self.sampler.sample(
                target_probablities[-1]
            )
            return VerificationResult(
                accepted_tokens=result.accepted_tokens,
                accepted_length=result.accepted_length,
                rejected_index=-1,
                replacement_token=None,
                bonus_token=bonus_token
            )
        return VerificationResult(
                accepted_tokens=result.accepted_tokens,
                accepted_length=result.accepted_length,
                rejected_index=result.rejected_index,
                replacement_token=result.replacement_token,
                bonus_token=None
            )