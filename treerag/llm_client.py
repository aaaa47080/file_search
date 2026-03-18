"""
llm_client.py
─────────────────────────────────────────────────────
LLM 包裝層，支援 OpenAI / Anthropic / Ollama（本地）
提供統一的 .chat(messages) -> str 介面
"""

import os
from typing import Optional


class LLMClient:
    """
    統一 LLM 介面
    支援：
      - openai: GPT-4o, GPT-4o-mini 等
      - anthropic: Claude 系列
      - ollama: 本地模型（Llama3, Mistral 等）
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        temperature: float = 0.1,
    ):
        """
        provider: "openai" | "anthropic" | "ollama"
        model: 模型名稱（不填則用預設）
        api_key: API 金鑰（也可用環境變數）
        base_url: 自定義端點（Ollama 或 proxy 使用）
        temperature: 0.0-1.0（建議用低值確保一致性）
        """
        self.provider = provider.lower()
        self.temperature = temperature

        # 設定預設模型
        defaults = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-5-haiku-20241022",
            "ollama": "llama3.2",
        }
        self.model = model or defaults.get(self.provider, "gpt-4o-mini")
        self.api_key = api_key or self._get_env_key()
        self.base_url = base_url

        self._client = None

    def _get_env_key(self) -> Optional[str]:
        """從環境變數取得 API Key"""
        env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "ollama": None,
        }
        env_var = env_map.get(self.provider)
        if env_var:
            return os.environ.get(env_var)
        return None

    def _get_client(self):
        """延遲初始化 client"""
        if self._client is not None:
            return self._client

        if self.provider == "openai":
            from openai import OpenAI
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)

        elif self.provider == "anthropic":
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.api_key)

        elif self.provider == "ollama":
            # Ollama 使用 OpenAI-compatible API
            from openai import OpenAI
            self._client = OpenAI(
                api_key="ollama",
                base_url=self.base_url or "http://localhost:11434/v1",
            )

        else:
            raise ValueError(f"不支援的 provider: {self.provider}")

        return self._client

    def chat(self, messages: list[dict], max_tokens: int = 2000) -> str:
        """
        主要介面：傳入 messages 列表，回傳回應文字

        messages 格式：
          [{"role": "user", "content": "..."}]
          [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        """
        client = self._get_client()

        try:
            if self.provider == "anthropic":
                return self._chat_anthropic(client, messages, max_tokens)
            else:
                return self._chat_openai(client, messages, max_tokens)
        except Exception as e:
            raise RuntimeError(f"LLM 呼叫失敗 ({self.provider}/{self.model}): {e}")

    def _chat_openai(self, client, messages: list, max_tokens: int) -> str:
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def _chat_anthropic(self, client, messages: list, max_tokens: int) -> str:
        # 分離 system message
        system_msg = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                user_messages.append(msg)

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": user_messages,
        }
        if system_msg:
            kwargs["system"] = system_msg

        response = client.messages.create(**kwargs)
        return response.content[0].text.strip()

    def test_connection(self) -> bool:
        """測試連線是否正常"""
        try:
            result = self.chat([{"role": "user", "content": "回應：OK"}], max_tokens=10)
            return "OK" in result or len(result) > 0
        except Exception as e:
            print(f"連線測試失敗：{e}")
            return False


def create_llm(provider: str = None, **kwargs) -> LLMClient:
    """
    便捷工廠函數

    用法：
      llm = create_llm("openai", model="gpt-4o-mini")
      llm = create_llm("anthropic", model="claude-3-5-haiku-20241022")
      llm = create_llm("ollama", model="llama3.2")

    也可以從環境變數自動偵測：
      export OPENAI_API_KEY=sk-...
      llm = create_llm()  # 自動用 openai

    預設行為：
      - 若未指定 provider，優先使用 OpenAI
      - 若無 OPENAI_API_KEY，才嘗試 Ollama
    """
    if provider is None:
        # 自動偵測：優先 OpenAI
        if os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        else:
            # 嘗試本地 Ollama
            provider = "ollama"

    return LLMClient(provider=provider, **kwargs)
