import os
import openai
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


def pydantic_usage():
    # We can define type enforced classes using pydantic
    # i.e., we have to adhere to these types otherwise it's a validation error at runtime
    class pUser(BaseModel):
        name: str
        age: int
        email: str

    foo_p = pUser(name="Jane", age=32, email="jane@gmail.com")
    print(foo_p.name)


def function_conversion():
    """
    We can use pydantic to define a function in python, and then convert it to the expected format (json) to be
    understandable by openai.
    :return:
    """

    # We define the function as a child of BaseModel. The docs are required for the function to be understood by the llm
    class WeatherSearch(BaseModel):
        """Call this with an airport code to get the weather at that airport"""
        airport_code: str = Field(description="airport code to get weather for")

    # We can use the converter to get the respective JSON
    weather_function = convert_to_openai_function(WeatherSearch)
    # If required, we can convert several functions:
    # functions = [
    #    convert_pydantic_to_openai_function(WeatherSearch),
    #    convert_pydantic_to_openai_function(ArtistSearch),
    # ]
    # We have to bind it to the model to be available automatically in the context
    model = ChatOpenAI(temperature=0).bind(
        functions=[weather_function])  # We can force calling via function_call={"name":"WeatherSearch"}
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant"),
        ("user", "{input}")
    ])
    chain = prompt | model
    print(chain.invoke({"input": "what is the weather in sf?"}))



if __name__ == '__main__':
    function_conversion()
