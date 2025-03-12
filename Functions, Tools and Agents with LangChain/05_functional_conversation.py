import os
import openai
from langchain.tools import tool
import requests
from pydantic import BaseModel, Field
import datetime
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain.schema.agent import AgentFinish
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents import AgentExecutor
from langchain.schema.runnable import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
import wikipedia

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


# We define the tools which shall be made available to the LLM
class OpenMeteoInput(BaseModel):
    # we can add descriptions for each field
    latitude: float = Field(..., description="Latitude of the location to fetch weather data for")
    longitude: float = Field(..., description="Longitude of the location to fetch weather data for")


# The typing is important for the model to understand the function.
# The args_schema is a pydantic schema which we use to annotate the function call
@tool(args_schema=OpenMeteoInput)
def get_current_temperature(latitude: float, longitude: float) -> dict:
    """Fetch current temperature for given coordinates."""
    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    # Parameters for the request
    params = {'latitude': latitude, 'longitude': longitude, 'hourly': 'temperature_2m', 'forecast_days': 1, }

    # Make the request
    response = requests.get(BASE_URL, params=params)

    if response.status_code == 200:
        results = response.json()
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")

    current_utc_time = datetime.datetime.now(datetime.UTC)
    current_utc_time = current_utc_time.replace(tzinfo=None)
    time_list = [datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00')) for time_str in
                 results['hourly']['time']]
    temperature_list = results['hourly']['temperature_2m']

    closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))
    current_temperature = temperature_list[closest_time_index]
    return f'The current temperature is {current_temperature}°C'


# We define a second "wikipedia" research tool
@tool
def search_wikipedia(query: str) -> str:
    """Run Wikipedia search and get page summaries."""
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[: 3]:
        try:
            wiki_page = wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except (self.wiki_client.exceptions.PageError, self.wiki_client.exceptions.DisambiguationError,):
            pass
    if not summaries:
        return "No good Wikipedia Search Result was found"
    return "\n\n".join(summaries)


# The list of all available tools as references
tools = [get_current_temperature, search_wikipedia]


def manual_loop():
    # We always have to convert the function declarations to the expected format
    functions = [convert_to_openai_function(f) for f in tools]
    model = ChatOpenAI(temperature=0).bind(functions=functions)
    prompt = ChatPromptTemplate.from_messages([("system", "You are helpful but sassy assistant"), ("user", "{input}"),
                                               MessagesPlaceholder(variable_name="agent_scratchpad")
                                               # The scratchpad is a placeholder where intermediate steps by the LLM (e.g., it has first to call a tool)
                                               # are added
                                               ])
    chain = prompt | model | OpenAIFunctionsAgentOutputParser()

    def run_agent(user_input):
        intermediate_steps = []
        while True:
            # format_to_openai_functions converts a function call and its result to one message
            result = chain.invoke(
                {"input": user_input, "agent_scratchpad": format_to_openai_functions(intermediate_steps)})
            # if we are in the finish state, just return
            if isinstance(result, AgentFinish):
                return result
            # if a tool is selected, use it
            tool = {"search_wikipedia": search_wikipedia, "get_current_temperature": get_current_temperature, }[
                result.tool]
            # execute the tool
            observation = tool.run(result.tool_input)
            # append the tool result and repeat the loop. result is hereby the tool call
            intermediate_steps.append((result, observation))

    print(run_agent("what is the weather in London"))
    print(run_agent("what is langgraph?"))

def automatic_loop():
    """
    Here we want instead to use the langchain agent executor which implements a loop and automatic tool execution and
    memory. The agent runs until an AgentFinish is reached.
    :return:
    """
    functions = [convert_to_openai_function(f) for f in tools]
    model = ChatOpenAI(temperature=0).bind(functions=functions)
    prompt = ChatPromptTemplate.from_messages([("system", "You are helpful but sassy assistant"),
                                               # the chat history stores the previous conversation --> memory
                                               MessagesPlaceholder(variable_name="chat_history"),
                                               ("user", "{input}"),
                                               MessagesPlaceholder(variable_name="agent_scratchpad")
                                               # The scratchpad is a placeholder where intermediate steps by the LLM (e.g., it has first to call a tool)
                                               # are added
                                               ])
    chain = prompt | model | OpenAIFunctionsAgentOutputParser()
    # On each iteration, add all intermediate steps to the scratchpad
    agent_chain = RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"])
    ) | chain
    # for the chat history we need memory, thus we initialize a buffer. we have to use the memory_key
    # according to the placeholder in the prompt template, only then the memory state replaces it in the prompt on each
    # request.
    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
    # the executor implements the loop
    agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True, memory=memory)
    print(agent_executor.invoke({"input": "what is the current weather in london?"}))

if __name__ == '__main__':
    print('Custom: ')
    manual_loop()
    print('Executor: ')
    automatic_loop()