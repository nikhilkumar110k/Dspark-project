import math
import random
from dataclasses import dataclass
from typing import Sequence

from dspark.residual_sample import (ResidualSampler,validate_distribution)

@dataclass(frozen=True)
class AcceptanceResult:
    accepted_tokens: tuple[int,...]
    accepted_length: int
    rejected_index: int
    replacement_index: int | None

    @property
    def all_accepted(self) -> bool:
        return self.rejected_index == -1

class Acceptance:
    def __init__(self,seed=None):
        self.rng= random.Random(seed)
        self.residual_sampler= ResidualSampler(rng=self.rng)

    @staticmethod
    def acceptance_probablity(target_probablity: float,draft_probablity: float)->float:
        if not math.isfinite(target_probablity):
            raise ValueError("target probablity cannot be infinite")
        if not math.isfinite(draft_probablity):
            raise ValueError("draft probablity cannot be infinite")
        if draft_probablity<=0.0:
            raise ValueError("draft probablity cannot be negative")
        if target_probablity<=0.0:
                    raise ValueError("target probablity cannot be negative")

        ratio= target_probablity/draft_probablity
        return min(1.0,ratio)
    
    def accept(self,alpha:float)->bool:
        if not 0.0<=alpha<=1:
              raise ValueError("alpha should be in the range from 0 to 1")
        random_value=self.rng.Random()
        return random_value<alpha
    
    @staticmethod 
    def find_accepted_prefix(acceptance_flags:Sequence[bool]) ->int:
        accepted_length=0

        for accepted in acceptance_flags:
            if not accepted:
                   break
            accepted_length += 1
        return accepted_length

    def verify_draft(self,draft_tokens: Sequence[int],draft_probablities: Sequence[Sequence[float]],target_probablities: Sequence[Sequence[float]]) -> AcceptanceResult:  
        draft_length= len(draft_tokens)
        if len(draft_probablities) != draft_length:
            raise ValueError("the length of draft probs and draft lengths don't match")
        if len(target_probablities) != draft_length:
            raise ValueError("the length of target probs and draft lengths don't match")
        accepted_tokens=[]
        
