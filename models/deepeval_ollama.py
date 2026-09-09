"""DeepEval adapter for a local Ollama judge model."""

import json

import ollama
from pydantic import BaseModel
from deepeval.models import DeepEvalBaseLLM

from config.model_config import JUDGE_MODEL


class OllamaEvaluator(DeepEvalBaseLLM):
    def __init__(self, model_name: str = JUDGE_MODEL):
        self.model_name = model_name

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str, schema: BaseModel = None):
        try:
            kwargs = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
            }
            if schema is not None:
                kwargs["format"] = schema.model_json_schema()
            response = ollama.chat(**kwargs)
            output = response["message"]["content"]
            if schema is not None:
                return schema.model_validate(json.loads(output))
            return output
        except Exception as exc:
            raise RuntimeError(
                f"Failed to evaluate with Ollama judge model '{self.model_name}'."
            ) from exc

    async def a_generate(self, prompt: str, schema: BaseModel = None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return f"Ollama {self.model_name}"
