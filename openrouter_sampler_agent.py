import os
import requests
from types import SimpleNamespace
from .eval_types import SamplerBase
import json
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT='''
- 你是一个 ReAct 风格的推理代理（agent），可以使用我为你提供的工具来回答用户问题，并且只可以用我提供的工具，不要自行调用其他外部工具以及你自带的默认工具，
- 请输出完整的思维链，包括思考过程、工具调用等。
- 如果调用工具，必须在思维链中记录选用该工具的原因以及它们的输入与输出（不能只写“使用了web_search”而不写具体输入/输出）。
- 在每次获得搜索结果后，必须评估：这些新信息是否足够回答用户的原始问题？如果答案已经明确，或者信息已经足够进行推断，**立即停止搜索**。
- 不要用相似的关键词反复搜索。如果连续两次搜索都没有获得有价值的新信息，说明该方向的信息可能已经穷尽，**必须停止搜索**并基于已有信息作答。
- 不要调用同一工具多于5次。如果已经调用了5次该工具，说明该工具可能无法提供更多有价值的信息，**必须停止使用该工具**并基于已有信息作答。
- 最后的三行（Explanation / Exact Answer / Confidence）**必须**精确按照格式输出，便于下游 grader 解析。
'''

from langchain.tools import tool
from .search_engine import GoogleSearchEngine

class WebSearchInput(BaseModel):
    query: str = Field(description="The search query to look up on the web.")
    num_results: int = Field(default=10, description="Number of search results to retrieve.")

def web_search(query: str, num_results: int = 10) -> str:
    """Google网络搜索工具。"""
    engine = GoogleSearchEngine()
    return engine.run(query, num_results)

# 注册工具时显式传入 args_schema
web_search_tool = StructuredTool.from_function(
    func=web_search,
    name="web_search_tool",
    description="Perform a web search using Google",
    args_schema=WebSearchInput, 
)

class OpenRouterSampler(SamplerBase):
    def __init__(self):
        # self.model = ChatOpenAI(
        #     model="z-ai/glm-4.5",
        #     openai_api_key = os.environ.get("ZHIPU_API_KEY"),
        #     openai_api_base="https://openrouter.ai/api/v1",
        # )
        self.model = ChatOpenAI(
            model="anthropic/claude-sonnet-4.5",
            openai_api_key = os.environ.get("CLAUDE_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
        )
        # self.checkpointer = InMemorySaver()
        self.agent = create_react_agent(
            model=self.model,
            tools=[web_search_tool],
            prompt=SYSTEM_PROMPT
        )
        # self.write_lock = threading.Lock()

    def _pack_message(self, content: str, role: str = "user"):
        return {"role": role, "content": content}


    def _extract_text(self, resp):
        if isinstance(resp, dict) and "messages" in resp:
            for msg in reversed(resp["messages"]):
                if msg.__class__.__name__ == "AIMessage":
                    return msg.content.strip()
        elif hasattr(resp, "content"):
            return resp.content.strip()
        return str(resp)
    
    def _extract_final_answer(self, text: str) -> str:
        start_idx = text.find("Final Answer")
        if start_idx == -1:
            return text 
        final_part = text[start_idx:]
        return final_part

    def __call__(self, messages):
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                resp = self.agent.invoke({"messages": messages},config={"configurable": {"max_iterations": 6}})
                text = self._extract_text(resp)
                if not resp or not text.strip():
                    raise ValueError("Empty response from model")
                
                with open('./result', 'a', encoding='utf-8') as f:
                    f.write("----------------------------------------------------------\n")
                    f.write("Ask And Response\n")
                    f.write("----------------------------------------------------------\n")
                    f.write("Messages:\n" + str(messages) + "\n")
                    f.write("----------------------------------------------------------\n")
                    f.write("Response:\n" + text + "\n")
                    f.write("----------------------------------------------------------\n\n")

                final_answer = self._extract_final_answer(text)

                return SimpleNamespace(
                    response_text=final_answer,
                    actual_queried_message_list=messages
                )
            
            except json.JSONDecodeError as e:
                last_error = e
                print(f"[Warning] JSONDecodeError: {e}. Attempt {attempt+1}/{max_retries}")
                continue

            except Exception as e:
                last_error = e
                print(f"[Error] Exception during agent.invoke: {e}")
                continue

        print("[Fatal] All attempts failed.")
        return SimpleNamespace(
            response_text=f"[ERROR] Agent failed after {max_retries} retries: {last_error}",
            actual_queried_message_list=messages
        )

class OpenRouterGrader(SamplerBase):
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

        with open('./result', 'a', encoding='utf-8') as f:
            f.write("----------------------------------------------------------\n")
            f.write("Grade\n")
            f.write("----------------------------------------------------------\n")
            f.write("Response:\n" + text + "\n")
            f.write("----------------------------------------------------------\n\n")

        return SimpleNamespace(
            response_text=text,
            actual_queried_message_list=messages
        )