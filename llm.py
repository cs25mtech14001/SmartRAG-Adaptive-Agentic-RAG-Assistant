"""Minimal wrapper around any OpenAI-compatible chat completion API."""
import config

SYSTEM_PROMPT = (
    "You are a precise assistant for computer networking questions. "
    "Be concise and factual. Follow the requested output format exactly."
)


class LLM:
    def __init__(self, model: str = config.LLM_MODEL, client=None):
        if client is None:
            from openai import OpenAI

            if not config.LLM_API_KEY:
                raise RuntimeError("LLM_API_KEY is not set. Copy .env.example to .env and add your key.")
            client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
        self.client = client
        self.model = model

    def __call__(self, prompt: str, max_tokens: int = 400) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()
