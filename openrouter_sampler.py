import os
import requests
from types import SimpleNamespace
from .eval_types import SamplerBase
import json

class OpenRouterSampler(SamplerBase):
    def __init__(self, api_key=None, model="z-ai/glm-4.5"):
        self.api_key = api_key or os.environ.get("ZHIPU_API_KEY")
        if not self.api_key:
            raise ValueError("Missing ZHIPU_API_KEY")
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"

    def _pack_message(self, content: str, role: str = "user"):
        return {"role": role, "content": content}

    def __call__(self, messages):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"         
        }
        payload = {
            "model": self.model,
            "messages": messages,
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            data=json.dumps(payload)
        )
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

# import os
# import requests
# from types import SimpleNamespace
# from .eval_types import SamplerBase
# from openai import OpenAI

# class OpenRouterSampler(SamplerBase):
#     def __init__(self,model="glm-4.5"):
#         self.model = model
#         self.api_key = "f47061767e828ad8f4c13250550db615.6og2kuaeCbujg5C7"
#         self.base_url="https://open.bigmodel.cn/api/paas/v4"


#     def _pack_message(self, content: str, role: str = "user"):
#         return {"role": role, "content": content}

#     def __call__(self, messages):
#         url = f"{self.base_url}/chat/completions"
#         headers = {"Authorization": f"Bearer {self.api_key}",
#                    "Content-Type": "application/json"}
#         data = {"model": self.model, "messages": messages}
#         resp = requests.post(url, headers=headers, json=data)
#         if resp.status_code != 200:
#             raise RuntimeError(f"Zhipu API error {resp.status_code}: {resp.text}")
#         output = resp.json()
#         text = output["choices"][0]["message"]["content"]
#         return SimpleNamespace(
#             response_text=text,
#             actual_queried_message_list=messages
#         )