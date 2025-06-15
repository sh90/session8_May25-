
from autogen.code_utils import create_virtual_env
from autogen.coding import CodeBlock, LocalCommandLineCodeExecutor
import data_info
configuration =  {"model": "gpt-4o-mini", "api_key": data_info.open_ai_key    }
venv_dir = ".envs1"
venv_context = create_virtual_env(venv_dir)

executor = LocalCommandLineCodeExecutor(
    virtual_env_context=venv_context,
    timeout=200,
    work_dir="coding",
)
print(
    executor.execute_code_blocks(code_blocks=[CodeBlock(language="python", code="import sys; print(sys.executable)")])
)

from autogen import AssistantAgent

# Agent that writes code
code_writer_agent = AssistantAgent(
    name="code_writer_agent",
    llm_config=configuration,
    code_execution_config=False,
    human_input_mode="NEVER",
)

# code_writer_agent_system_message = code_writer_agent.system_message
# print(code_writer_agent_system_message)

from autogen import ConversableAgent

# Code executor agent
code_executor_agent = ConversableAgent(
    name="code_executor_agent",
    llm_config=False,
    code_execution_config={"executor": executor},
    human_input_mode="ALWAYS",
    default_auto_reply=
    "Please continue. If everything is done, reply 'TERMINATE'.",
)

import datetime

today = datetime.datetime.now().date()

message = f"Today is {today}. "\
"Create a plot showing the normalized price of NVDA and BTC-USD for the last 4 years "\
"with their 50 weeks moving average. "\
"Make sure the code is in a markdown code block, save the figure"\
" to a file asset_analysis.png and show it. Provide all the code necessary in a single python bloc. "\
"Re-provide the code block that needs to be executed with each of your messages. "\
"If python packages are necessary to execute the code, provide a markdown "\
"sh block with only the command necessary to install them and no comments."

chat_result = code_executor_agent.initiate_chat(
    code_writer_agent,
    message=message
)
