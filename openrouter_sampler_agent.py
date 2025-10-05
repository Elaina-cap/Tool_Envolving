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

SYSTEM_PROMPT='''
你是一个 ReAct 风格的推理代理（agent），可以使用我为你提供提供的工具来回答用户问题，并且只可以用我提供的工具，不要自行调用外部工具，如果工具列表为空，请不要使用任何工具，单纯作为模型回答问题
**不要**输出内部的逐字思维过程（chain-of-thought）。你必须以对外可见、结构化的形式记录你的**行动与工具使用情况**，并给出**简洁的公开推理摘要**和最终答案。

严格遵循下面格式输出：

1) 【Action Log — 行动日志】（按时间顺序编号）
   每个行动项必须包含：
   - Action #: 从 1 开始的序号
   - Action Type: 取值之一：TOOL_CALL / TOOL_RESULT / COMPUTE / ANSWER
   - Tool: 使用的工具名（若非工具调用则写 "-"）
   - Input: 传入工具或执行的具体输入（或简短说明）
   - Output / Observation: 工具返回或本次操作的结果（若无则写 "-"）

   示例条目：
   1) Action #: 1
      Action Type: TOOL_CALL
      Tool: web_search
      Input: "MMA event loser 14 significant strikes 83 attempted nickname swordsman"
      Output / Observation: "搜索结果摘要：...（给出工具返回的原文摘要或片段）"

   对工具返回的每个重要片段，必须把原文或精确摘要写入 Output / Observation，以便审计。

2) 【Public Reasoning — 公开推理摘要】（用简短句子描述每一步“为什么”）
   - 只写对外摘要（每条 1-2 句），不要写内部思路或逐词的链式推理。
   - 每条要能被外部审阅者理解为什么采取了对应的行动（例如“因为需要确认比赛日期与统计数据，所以用 web_search”）。
   - 对应 Action Log 中的关键行动请在括号里标注该 action 的序号，便于交叉核验。

   示例：
   - Step 1: 为确认选手绰号含义，检索相关比赛与选手简介（参见 Action #1 的 web_search）。  
   - Step 2: 基于工具返回的赛事统计摘要，筛选满足“14/83、0 次摔跤”条件的赛事（参见 Action #2 的 search result）。

3) 【Sources / Evidence — 证据/来源】  
   - 列出所有引用的工具输出、URL 或文档片段，并标注对应 Action #（例如：Action #1 -> https://example.com/page）。  
   - 如果工具输出为原文片段，请直接粘贴短片段（≤200 字），并注明来源 URL。

4) 【Final Answer — 最终答案（必须包含三行，严格格式）】
   Explanation: {在此处写 2-4 句的、非内部的、可审计的推理总结 — 不要写逐字思考}
   Exact Answer: {简洁的、可核验的最终答案（事件名 / 日期 / 参照字段等）}
   Confidence: {0% - 100% 的置信度估计，若不确定则给范围并说明原因}

5) 若无法确认或证据不足，请明确写：
   - "Unable to confidently answer — missing evidence" 并列出缺失的关键证据项（例如：确切比赛报告、官方统计页面等），以及下一步应检索的具体查询字符串（便于自动/人工跟进）。

补充规则（必须遵守）：
- 绝不编造来源或伪造数据；如引用来自工具的内容，请在 Action Log 中记录对应工具的完整输出片段或链接。
- 公开推理摘要应简洁且可验证（可用 1-3 句说明每个重要决策点）。
- 如果调用多个工具，必须在 Action Log 中记录它们的输入与输出（不能只写“使用了 web_search”而不写具体输入/输出）。
- 输出长度：Action Log / Evidence 可较长，但 Public Reasoning 与 Final Answer 要简洁（每条 ≤ 2 句）。
- 最后的三行（Explanation / Exact Answer / Confidence）**必须**精确按照格式输出，便于下游 grader 解析。


示例（简短）输出片段（仅示例格式）：
Action Log:
1) Action #:1
   Action Type: TOOL_CALL
   Tool: web_search
   Input: "MMA event loser 14 significant strikes 83 attempted nickname swordsman"
   Output / Observation: "Result A: http://... 'Fighter X (nickname Swordsman) lost at Event Y; stats: 14/83, 0 TD'"

Public Reasoning:
- Step 1: 查询绰号与统计，以定位可能的比赛（参见 Action #1）。
- Step 2: 验证比赛日期与裁判记录（参见 Action #2 的裁判页面）。

Sources / Evidence:
- Action #1 -> http://... (摘录: '14 significant strikes of 83 attempted...')

Final Answer:
Explanation: 基于 web_search 的比赛记录与官方统计页面（见 Sources）匹配题目所有条件，因此答案定位为 Event Y。
Exact Answer: Event Y
Confidence: 85%
'''

from langchain.tools import tool
from .search_engine import DDGSSearchEngine
# @tool("web_search", return_direct=False)
# def web_search(query: str, num_results: int = 10) -> str:
#     """DuckDuckGo 网络搜索工具。"""
#     schema={
#         "name":"web_search",
#         "description" :"a web search tool",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "query": {
#                     "type": "string",
#                     "description": "The search query to look up on the web."
#                 },
#                 "num_results": {
#                     "type": "integer",
#                     "description": "Number of search results to retrieve (default 10).",
#                     "default": 10,
#                 }
#             },
#             "required": ["query"]
#         },
#         "output": {
#             "type": "object",
#             "properties": {
#                 "results": {
#                     "type": "array",
#                     "description": "A list of search results (each containing title, url, and snippet).",
#                     "items": {
#                         "type": "object",
#                         "properties": {
#                             "title": {"type": "string", "description": "Title of the web page."},
#                             "url": {"type": "string", "description": "Link to the web page."},
#                             "description": {"type": "string", "description": "Short text snippet or summary."}
#                         }
#                     }
#                 },
#             }
#         }
#     }
#     try:
#         searcher = DDGSSearchEngine()
#         return searcher.run(query,num_results)
#     except Exception as e:
#         return f"[web_search error] {e}"



class WebSearchInput(BaseModel):
    query: str = Field(description="The search query to look up on the web.")
    num_results: int = Field(default=10, description="Number of search results to retrieve.")

def web_search(query: str, num_results: int = 10) -> str:
    """DuckDuckGo 网络搜索工具。"""
    engine = DDGSSearchEngine()
    return engine.run(query, num_results)

# 注册工具时显式传入 args_schema
web_search_tool = StructuredTool.from_function(
    func=web_search,
    name="web_search",
    description="Perform a web search using DuckDuckGo",
    args_schema=WebSearchInput, 
)

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
            tools=[web_search],
            prompt=SYSTEM_PROMPT,
        )
        if hasattr(self.agent, "tools"):
            self.agent.tools = []

        if hasattr(self.agent, "tool_executor"):
            self.agent.tool_executor.tools = {}
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
                resp = self.agent.invoke({"messages": messages})
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
            f.write("Messages:\n" + str(messages) + "\n")
            f.write("----------------------------------------------------------\n")
            f.write("Response:\n" + text + "\n")
            f.write("----------------------------------------------------------\n\n")

        return SimpleNamespace(
            response_text=text,
            actual_queried_message_list=messages
        )