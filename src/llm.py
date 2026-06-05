from __future__ import annotations

from typing import Callable
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class LLM:
    def __init__(self, model: str = "cx/gpt-5.5", provider: str = "openai"):
        self.model = model
        self.client = None
        if provider == "openai":
            self.client = OpenAI(
                base_url=os.getenv("OPENAI_API_BASE_URL", "http://localhost:20128/v1/"),
                api_key=os.getenv("OPENAI_API_KEY", ""),
            )

    def generate(self, prompt: str) -> str:
        if self.client:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()
        else:
            return f"[MOCK LLM] Echoing prompt: {prompt[:100]}..."