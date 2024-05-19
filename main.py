import os
from typing import Union, List, Tuple

from langchain.agents import tool
from langchain.prompts.prompt import PromptTemplate
from langchain.tools.render import render_text_description
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.tools import Tool

from langchain_openai import AzureChatOpenAI

from callbacks import AgentCallbackHandler # para imprimir los pasos intermedios

@tool
def get_text_length(text:str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip(
        '"'
    ) # stripping away non alphabetic characters just in case
    return len(text)

def find_tool_by_name(tools: List[Tool], tool_name:str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")

def format_log_to_str(
        intermediate_steps: List[Tuple[AgentAction, str]],
        observation_prefix: str = "Observation: ",
        llm_prefix: str = "Thought: ",
) -> str:
    """Construct the scratchpad that lets the agent continue its thought process."""
    thoughts = ""
    for action, observation in intermediate_steps:
        thoughts += action.log
        thoughts += f"\n{observation_prefix}{observation}\n{llm_prefix}"
    return thoughts

if __name__ == "__main__":
    print("Hello ReAct LangChain!")
    tools = [get_text_length]

    template="""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(       # usamos partial para rellenar sólo un subset de los valores del prompt
        tools=render_text_description(tools=tools),     # no pasamos las tools directamente porque necesitamos convertirlas a texto para el LLM
        tool_names=", ".join([t.name for t in tools])
        )
    
    # AZURE OPENAI
    llm = AzureChatOpenAI(
        azure_deployment = os.environ.get('DEPLOYMENT_NAME'),
        temperature = 0,
        model_kwargs = {"stop":["\nObservation", "Observation"]},      # kwargs = keyword arguments. Pasamos stop words para detener la generación de texto
        callbacks = [AgentCallbackHandler()]
    )

    intermediate_steps = []

    agent = (
        {
            "input": lambda x:x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    agent_step = ""
    while not isinstance(agent_step, AgentFinish):     
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the length in characters of the text DOG?",
                "agent_scratchpad": intermediate_steps,
            }
        )

        print(agent_step)

        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input

            observation = tool_to_use.func(str(tool_input))
            print(f"{observation}")
            intermediate_steps.append((agent_step, str(observation)))

    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)