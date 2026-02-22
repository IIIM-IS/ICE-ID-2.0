"""
MatchGPT model adapter.

Wrapper for MatchGPT - LLM-based entity matching using prompts.

Reference: https://github.com/wbsg-uni-mannheim/MatchGPT
Papers: "Using ChatGPT for Entity Matching" and "Entity Matching using Large Language Models"
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from .base import BaseModel
from ..core.types import DatasetSplit, Pair
from ..core.registry import get_registry


class MatchGPTModel(BaseModel):
    """
    MatchGPT model adapter.
    
    Uses LLM prompts (OpenAI GPT, etc.) for entity matching.
    Requires OPENAI_API_KEY environment variable to be set.
    """
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        max_tokens: int = 10,
        prompt_template: str = "default",
        use_few_shot: bool = True,
        n_shots: int = 3,
        **kwargs
    ):
        """
        Initialize MatchGPT model.
        
        Args:
            model: OpenAI model name (gpt-3.5-turbo, gpt-4, etc.).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            prompt_template: Prompt template to use.
            use_few_shot: Whether to use few-shot prompting.
            n_shots: Number of examples in few-shot prompt.
        """
        if not HAS_OPENAI:
            raise ImportError("OpenAI not installed. Run: pip install openai")
        
        super().__init__("matchgpt", **kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_template = prompt_template
        self.use_few_shot = use_few_shot
        self.n_shots = n_shots
        self.few_shot_examples = []
        
        if not os.getenv("OPENAI_API_KEY"):
            print("Warning: OPENAI_API_KEY not set. MatchGPT will use dummy scores.")
    
    def _create_prompt(self, rec1: Dict, rec2: Dict) -> str:
        """Create prompt for entity matching."""
        if self.prompt_template == "default":
            prompt = "Compare these two records and determine if they refer to the same entity.\\n\\n"
            prompt += "Record 1:\\n"
            for key, val in rec1.items():
                if key != "id":
                    prompt += f"  {key}: {val}\\n"
            
            prompt += "\\nRecord 2:\\n"
            for key, val in rec2.items():
                if key != "id":
                    prompt += f"  {key}: {val}\\n"
            
            prompt += "\\nAre these the same entity? Answer with 'YES' or 'NO'."
            
            return prompt
        
        return f"Record 1: {rec1}\\nRecord 2: {rec2}\\nMatch?"
    
    def fit(
        self,
        dataset: DatasetSplit,
        train_pairs: pd.DataFrame,
        val_pairs: Optional[pd.DataFrame] = None,
    ):
        """
        Store few-shot examples from training data.
        
        MatchGPT doesn't train; it uses few-shot prompting.
        """
        if self.use_few_shot and train_pairs is not None:
            id1_col = "id1" if "id1" in train_pairs.columns else "ltable_id"
            id2_col = "id2" if "id2" in train_pairs.columns else "rtable_id"
            
            positives = train_pairs[train_pairs["label"] == 1].head(self.n_shots // 2)
            negatives = train_pairs[train_pairs["label"] == 0].head(self.n_shots // 2 + 1)
            
            examples = pd.concat([positives, negatives])
            
            for _, row in examples.iterrows():
                id1 = int(row[id1_col])
                id2 = int(row[id2_col])
                label = int(row["label"])
                
                rec1 = dataset.get_record_by_id(id1) or {}
                rec2 = dataset.get_record_by_id(id2) or {}
                
                self.few_shot_examples.append({
                    "rec1": rec1,
                    "rec2": rec2,
                    "label": label
                })
        
        self._is_fitted = True
    
    def score(
        self,
        dataset: DatasetSplit,
        pairs: List[Pair],
    ) -> np.ndarray:
        """Score pairs using LLM prompts."""
        if not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY not set. Returning random scores.")
            return np.random.rand(len(pairs)) * 0.5 + 0.5
        
        scores = []
        
        for id1, id2 in pairs:
            rec1 = dataset.get_record_by_id(id1) or {}
            rec2 = dataset.get_record_by_id(id2) or {}
            
            messages = []
            
            if self.use_few_shot and self.few_shot_examples:
                system_msg = "You are an expert at entity matching. "
                system_msg += "Given two records, determine if they refer to the same entity."
                messages.append({"role": "system", "content": system_msg})
                
                for example in self.few_shot_examples:
                    prompt = self._create_prompt(example["rec1"], example["rec2"])
                    answer = "YES" if example["label"] == 1 else "NO"
                    messages.append({"role": "user", "content": prompt})
                    messages.append({"role": "assistant", "content": answer})
            
            prompt = self._create_prompt(rec1, rec2)
            messages.append({"role": "user", "content": prompt})
            
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    n=1,
                )
                
                answer = response.choices[0].message.content.strip().upper()
                
                if "YES" in answer:
                    scores.append(0.9)
                elif "NO" in answer:
                    scores.append(0.1)
                else:
                    scores.append(0.5)
                
            except Exception as e:
                print(f"MatchGPT API error: {e}")
                scores.append(0.5)
        
        return np.array(scores)


if HAS_OPENAI:
    get_registry("models").register("matchgpt", MatchGPTModel)


