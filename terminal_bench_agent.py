import os
import ast
import time
from pathlib import Path
from types import SimpleNamespace
from pydantic import BaseModel, Field

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool

from terminal_bench.agents.base_agent import BaseAgent,AgentResult
from terminal_bench.terminal.tmux_session import TmuxSession
from terminal_bench.agents.failure_mode import FailureMode

# def web_search(query: str, num_results: int = 5) -> str:
#     """DuckDuckGo 网络搜索工具。"""
#     try:
#         from simple_evals.search_engine import DDGSSearchEngine
#         engine = DDGSSearchEngine()
#         return engine.run(query, num_results)
#     except Exception as e:
#         return f"[web_search error] {e}"


# class WebSearchInput(BaseModel):
#     query: str = Field(description="The search query to look up on the web.")
#     num_results: int = Field(default=5, description="Number of search results to retrieve.")


# web_search_tool = StructuredTool.from_function(
#     func=web_search,
#     name="web_search",
#     description="Perform a web search using DuckDuckGo",
#     args_schema=WebSearchInput,
# )


SYSTEM_PROMPT = """
你是一个具备工具调用能力的智能代理，可以执行命令行任务。
如果任务描述涉及命令执行，请输出相应的 shell 命令。
你输出的命令需要经过运行测评，所以最后一条指令务必以换行符结尾，这样才能在终端中运行。
当确定答案后，请严格按照 “Final Answer:需要执行的指令（不要包含任何多余的说明文字,并且请以list[str]列表的格式输出所有指令）” 的格式说明最终答案。
"""

class YourCustomAgent(BaseAgent):
    """可直接用于 terminal-bench 的自定义 Agent"""

    @staticmethod
    def name() -> str:
        return "your-agent-name"

    def __init__(self, *args, **kwargs):
        self.no_rebuild = kwargs.get("no_rebuild", False)
        self.model = ChatOpenAI(
            model="z-ai/glm-4.5",
            openai_api_key=os.environ.get("ZHIPU_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
        )

        self.agent = create_react_agent(
            model=self.model,
            tools=[],
            prompt=SYSTEM_PROMPT + "\n\n请在得出最终结果后停止。（Final Answer:需要执行的指令列表） 为停止信号。",
        )
        

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        
        print(f"[YourCustomAgent] Running task: {instruction}")

        try:
            resp = self.agent.invoke({"messages": [{"role": "user", "content": instruction}]})
            print(resp)
            output_text = self._extract_text(resp)
            final_command=self._extract_final_answer(output_text)
            final_command_list= ast.literal_eval(final_command)
            final_command_list = [cmd if cmd.endswith("\n") else cmd + "\n" for cmd in final_command_list]
            print(final_command_list)
            token_used = self._extract_token(resp)
        except Exception as e:
            output_text = f"[ERROR] {e}"

        if session:
            try:
                session.send_keys(final_command_list, block=False)
                time.sleep(1)
                terminal_output = session.get_incremental_output()
                print("Terminal Output:\n", terminal_output)
            except Exception as e:
                terminal_output = f"[ERROR running command] {e}"
        else:
            terminal_output = "[No session provided]"
        print("Terminal Output:\n", terminal_output)

        if logging_dir:
            os.makedirs(logging_dir, exist_ok=True)
            with open(Path(logging_dir) / "agent_output.txt", "w", encoding="utf-8") as f:
                f.write(f"MODEL OUTPUT:\n{output_text}\n\nTERMINAL OUTPUT:\n{terminal_output}")

        return AgentResult(
            total_input_tokens=token_used['input_tokens'],
            total_output_tokens=token_used['output_tokens'],
            failure_mode=FailureMode.NONE,
            timestamped_markers=[(session.get_asciinema_timestamp(), "Command executed")],
        )

    def _extract_text(self, resp):
        if isinstance(resp, dict) and "messages" in resp:
            for msg in reversed(resp["messages"]):
                if msg.__class__.__name__ == "AIMessage":
                    return msg.content.strip()
        elif hasattr(resp, "content"):
            return resp.content.strip()
        return str(resp)
    
    def _extract_token(self, resp):
        if isinstance(resp, dict) and "messages" in resp:
            for msg in reversed(resp["messages"]):
                if msg.__class__.__name__ == "AIMessage":
                    return msg.usage_metadata

    def _extract_final_answer(self, text: str) -> str:
        marker = "Final Answer:"
        start_idx = text.find(marker)
        if start_idx == -1:
            return text.strip()
        return text[start_idx + len(marker):].strip()
    
if __name__ == "__main__":
    agent = YourCustomAgent()

    # 模拟一个假的 session（不会真的执行 tmux 命令）
    session = SimpleNamespace(
        send_command=lambda cmd: print(f"[tmux mock send] {cmd}"),
        read_output=lambda: "hello world"
    )

    result = agent.perform_task(
        "Print hello world to the terminal.",
        session,
        logging_dir=Path("./logs")
    )
    print(result)

#   tb run \
#   --dataset-path ~/terminal-bench/tasks \
#   --agent-import-path simple_evals.terminal_bench_agent:YourCustomAgent \
#   --task-id hello-world
