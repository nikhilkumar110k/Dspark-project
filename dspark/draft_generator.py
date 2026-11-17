import math
import random

from dataclasses import dataclass
from typing import Protocol, Sequence

class DraftModel(Protocol):
    def next_token_logits(self,context: Sequence[int])->Sequence[float]:
        ...

@dataclass(frozen=True)
class DraftResult:
    tokens: tuple[int,...]
    probablities: tuple[tuple[float,...],...]
    sampled_probablities: tuple[float,...]

class DraftGenerator:
    def __init__(self,model: DraftModel,temperature:float = 1.0,seed=None):
        if not math.isfinite(temperature) or temperature<=0:
            raise ValueError("the temperature should be a finite number or greater than zero")
        self.model=model
        self.temperature=temperature
        self.rng=random.Random(seed)

    def softmax(self,logits:Sequence[float])->list[float]:
        if len(logits)==0:
            raise ValueError("logits cannot be empty")

        for value in logits:
            if not math.isfinite(value):
                raise ValueError("the logits values should be finite")
        scaled_logits= [value/self.temperature for value in logits]
        max_logit= max(scaled_logits)
        exponentials=[math.exp(value-max_logit) for value in logits]
        total=sum(exponentials)
        probablities= [value/total for value in exponentials]
        return probablities

    def sample_token(self,probablities: Sequence[float])->int:
        if len(probablities)==0:
            raise ValueError("probablities cannot be empty")
        if any(not math.isfinite(value) or value<0.0 for value in probablities):
            raise ValueError("the probablties should be finite or greater than zero")
        total=sum(probablities)
        if not math.isclose(total,1.0,abs_tol=1e-9):
            raise ValueError("the probablities sum should be 1")
        
        u=self.rng.random()
        cumulative=0.0

        for token_id, probablity in enumerate(probablities):
            cumulative+=probablity
            if u<cumulative:
                return token_id
        for token_id in range(len(probablities)-1,-1,-1):
            if probablities[token_id] >0.0:
                return token_id
        raise ValueError("no token has positive probablity")
    
    def generate(self,prompt_tokens: Sequence[int],max_tokens:int)-> DraftResult:
        if len(prompt_tokens)==0:
            raise ValueError("the prompt tokens are empty")
        if not isinstance(max_tokens,int) or max_tokens<=0:
            raise ValueError("the max token must be a positive interger")
        context=list(prompt_tokens)
        generated_tokens=[]
        all_probablities=[]
        sampled_probablities=[]
        for _ in range(max_tokens):
            logits= self.model.next_token_logits(context)
            probablities=self.softmax(logits)
            token=self.sample_token(probablities)
            generated_tokens.append(token)
            all_probablities.append(tuple(probablities))
            sampled_probablities.append(probablities[token])
            context.append(token)
        return DraftResult(
            tokens=tuple(generated_tokens),
            probablities=tuple(all_probablities),
            sampled_probablities=tuple(sampled_probablities)
        )