import math
import random 
from typing import Sequence

def validate_distribution(distribution: Sequence[float], tolerance: float = 1e-9) -> None:
    if len(distribution) == 0:
            raise ValueError("distribution is empty")
    total = 0.0
    for probablity in distribution:
          if not math.isfinite(probablity):
                raise ValueError("probablities are infinite")
          if probablity < 0.0:
                raise ValueError("probablities cannot be in negative")
          total +=probablity
    if not math.isclose(total,1.0,rel_tol=tolerance,abs_tol=tolerance):
          raise ValueError("the distributio is not valid cause the sum is not 1")


class ResidualSampler:          
      def __init__(self,rng=None):
            self.rng= rng if rng is not None else random.Random()

      def calculate_residual(self,target_distributon: Sequence[float],draft_distrbuton: Sequence[float]) -> list[float]:
            validate_distribution(target_distributon)
            validate_distribution(draft_distrbuton)
            if len(target_distributon) != len(draft_distrbuton):
                  raise ValueError("the length of the both distribution should be same")
            residual=[]

            for p,q in zip(target_distributon,draft_distrbuton):
                  value= p- min(p,q)
                  residual.append(max(0.0,value))
            return residual
      
      def normalize(self,residual: Sequence[float])->list[float]:
            total=sum(residual)
            if(total<=0.0):
                  raise ValueError("residual is zero is in negative")
            return [value/total for value in residual] 
      
      def sample(self,distribution: Sequence[float])-> int:
            validate_distribution(distribution)
            random_value=self.rng.random()
            cumulative=0.0
            last_positive_index= None

            for index,probablity in enumerate(distribution):
                if(probablity>0.0):
                        last_positive_index=probablity
                cumulative+=probablity
                if random_value<cumulative:
                      return index
            if last_positive_index is not None:
                  return last_positive_index
            raise ValueError("cannot sample empty distribution")

      
      def sample_replacement(self,target_distributon,draft_distrbuton) ->int:
            residual=self.calculate_residual(target_distributon,draft_distrbuton)
            normalized_residual=self.normalize(residual)

            return self.sample(normalized_residual)