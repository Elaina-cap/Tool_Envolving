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
from .search_engine import GoogleSearchEngine
from .docker_code_runner import DockerCodeExecutorTool

NOW_TASK_IMAGE="tb-conda-env-conflict-resolution-pre"

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

class DockerCodeInput(BaseModel):
    code: list[str] = Field(description= """A list of bash commands to be executed.
            Each item in the list must be a string representing a single bash command.
            For example, ['echo \"Hello, World!\"', 'ls -l']. 
            Make sure the list contains valid bash commands as strings.""")

def docker_code(code: list[str]):
    executor = DockerCodeExecutorTool()
    code = "\n".join(code)
    return executor.run(code,NOW_TASK_IMAGE)

# 注册工具时显式传入 args_schema
docker_code_tool = StructuredTool.from_function(
    func=docker_code,
    name="docker_code_tool",
    description="""You can use this tool to test whether your code is correct. 
    Your code will be executed in a docker environment and then you can observe the result 
    """,
    args_schema=DockerCodeInput, 
)

SYSTEM_PROMPT="""
You are a professional command-line task execution assistant with powerful tool-calling capabilities.

Core Responsibilities:
- Efficiently execute various tasks within terminal-bench.
- Provide precise shell commands for command execution tasks.
- You are currently running in a container environment with root privileges.

Workflow:
- Carefully analyze the task requirements.
- Plan the execution steps and tool calls.
- Output the complete thought process (including tool selection, execution logic, etc.).
- Provide the final execution command.

Output Format Requirements:
- Thought Process: Detail the analysis steps and decision logic.
- Final Answer: Strictly follow the “Final Answer:command_list” format.
- Command Format: Must be of type list[str] and must not include sudo.
- Please stop after outputting the complete thought chain and the final result. “Final Answer:…” serves as the stop signal.

Important Notes:
- Prioritize the safety and efficiency of command execution.
- Ensure all commands can run properly in the container environment.
- Keep the output concise and accurate.
- Adhere to standard shell syntax; do not include extraneous explanations.
"""
# SYSTEM_PROMPT = """
# 你是一个专业的命令行任务执行助手，具备强大的工具调用能力。

# 核心职责：
# - 高效执行terminal-bench中的各类任务
# - 为命令执行任务提供精确的shell命令
# - 你当前以root权限运行在容器环境中

# 工作流程：
# 1. 仔细分析任务需求
# 2. 规划执行步骤和工具调用
# 3. 完整输出思考过程（包括工具选择、执行逻辑等）
# 4. 提供最终的执行指令

# 输出格式要求：
# - 思考过程：详细说明分析步骤和决策逻辑
# - 最终答案：严格按 "Final Answer:指令列表" 格式输出，这里“Final Answer:”一定要用英文
# - 指令格式：list[str]类型，不包含sudo
# - 请在输出完整思维链并得出最终结果后停止。（Final Answer:需要执行的指令列表） 为停止信号。

# 注意事项：
# - 优先考虑命令执行的安全性和效率
# - 确保指令在容器环境中可正常运行
# - 输出保持简洁准确
# - 要符合shell语法规范，不要包含多余解释
# """

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
            prompt=SYSTEM_PROMPT,
        )
        

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        
        try:
            print("-----------------------------------------------------------------")
            resp = self.agent.invoke({"messages": [{"role": "user", "content": instruction}]})
            print(resp)
            print("-----------------------------------------------------------------")
            output_text = self._extract_text(resp)
            print(output_text)
            print("-----------------------------------------------------------------")
            final_command=self._extract_final_answer(output_text)
            final_command_list= ast.literal_eval(final_command)
            final_command_list = [cmd if cmd.endswith("\n") else cmd + "\n" for cmd in final_command_list]
            # print(final_command_list)
            # print("-----------------------------------------------------------------")
        except Exception as e:
            output_text = f"[ERROR] {e}"

        if session:
            try:
                session.send_keys(final_command_list, block=True)
                time.sleep(1)
                terminal_output = session.get_incremental_output()
            except Exception as e:
                terminal_output = f"[ERROR running command] {e}"
        else:
            terminal_output = "[No session provided]"

        if logging_dir:
            os.makedirs(logging_dir, exist_ok=True)
            with open(Path(logging_dir) / "agent_output.txt", "w", encoding="utf-8") as f:
                f.write(f"MODEL OUTPUT:\n{output_text}\n\nTERMINAL OUTPUT:\n{terminal_output}")

        return AgentResult(
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
                    if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                        return msg.usage_metadata
            return {"input_tokens":0,"output_tokens":0}

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
#   --agent-import-path Tool_Envolving.terminal_bench_agent:YourCustomAgent \
#   --task-id hello-world

#   tb run \
#   --dataset-path ~/terminal-bench/tasks \
#   --agent-import-path Tool_Envolving.terminal_bench_agent:YourCustomAgent \
#   --task-id broken-networking

#   tb run \
#   --dataset-path ~/terminal-bench/tasks \
#   --agent-import-path Tool_Envolving.terminal_bench_agent:YourCustomAgent \
#   --task-id git-multibranch

#   tb run \
#   --dataset-path ~/terminal-bench/tasks \
#   --agent-import-path Tool_Envolving.terminal_bench_agent:YourCustomAgent \
#   --task-id crack-7z-hash

#   tb run \
#   --dataset-path ~/terminal-bench/tasks \
#   --agent-import-path Tool_Envolving.terminal_bench_agent:YourCustomAgent \
#   --task-id conda-env-conflict-resolution


#   tb run \
#   --dataset-path ~/terminal-bench/tasks \
#   --agent-import-path Tool_Envolving.terminal_bench_agent:YourCustomAgent \
#   --task-id eval-mteb

#   tb run \
#   --dataset-path ~/terminal-bench/tasks \
#   --agent-import-path Tool_Envolving.terminal_bench_agent:YourCustomAgent \
#   --task-id build-tcc-qemu


#   tb run \
#   --dataset-path ~/terminal-bench/tasks \
#   --agent-import-path Tool_Envolving.terminal_bench_agent:YourCustomAgent \
#   --task-id extract-safely