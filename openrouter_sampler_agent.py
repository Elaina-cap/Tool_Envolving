import os
import requests
from types import SimpleNamespace
from .eval_types import SamplerBase
import json
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

class OpenRouterSampler(SamplerBase):
    def __init__(self):
        self.model = ChatOpenAI(
            model="z-ai/glm-4.5",
            openai_api_key = os.environ.get("ZHIPU_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
        )
        # self.checkpointer = InMemorySaver()
        self.agent = create_react_agent(
            model=self.model,
            tools=[], 
            prompt="You are a helpful assistant",
            # checkpointer=self.checkpointer
        )

    def _pack_message(self, content: str, role: str = "user"):
        return {"role": role, "content": content}

    def __call__(self, messages):
        print("----------------------------------------------------------")
        print(messages)
        print("----------------------------------------------------------")
        resp = self.agent.invoke({'messages': [{'role': 'user', 'content': messages}]})
        print("----------------------------------------------------------")
        print(resp)
        print("----------------------------------------------------------")
        if resp.status_code != 200:
            raise RuntimeError(f"OpenRouter API error {resp.status_code}: {resp.text}")
        try:
            data = resp.json()
        except Exception:
            print("Response is not JSON:", resp.text) 
            raise
        text = data["choices"][0]["message"]["content"]

        return SimpleNamespace(
            response_text=text,
            actual_queried_message_list=messages
        )

