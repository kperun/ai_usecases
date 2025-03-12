import os
from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.schema.runnable import RunnableMap
import json

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file

def simple_chain():
    """
    A simple chain consists of an input, a processing and an output parsing
    :return:
    """
    prompt = ChatPromptTemplate.from_template(
        "tell me a short joke about {topic}"
    )
    model = ChatOpenAI()
    output_parser = StrOutputParser() #parses the model response to a string
    # the chain is created using the pipe symbol
    chain = prompt | model | output_parser
    print(chain.invoke({"topic": "bears"}))

def complex_chain():
    """
    We want an additional RAG step before the prompt to provide context to work on.
    :return:
    """
    # This is a fake vector store, can be replaced by a real one
    vectorstore = DocArrayInMemorySearch.from_texts(
        ["harrison worked at kensho", "bears like to eat honey"],
        embedding=OpenAIEmbeddings() # we are using OpenAi, thus have to use the same embeddings
    )
    retriever = vectorstore.as_retriever()
    # We create a template with a context part and a question part
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI()
    output_parser = StrOutputParser()  # parses the model response to a string
    # the runnable map can split one question into the context retrieval and the question again.
    # its basically a custom step in the pipe
    chain = RunnableMap({
        "context": lambda x: retriever.invoke(x["question"]),# We use a store to retrieve relevant documents
        "question": lambda x: x["question"]
    }) | prompt | model | output_parser
    # We start the chain via invoke (or batch or stream)
    print(chain.invoke({"question": "where did harrison work?"}))

def function_binding():
    """
    Here we bind a function to the model so that it is able to tell us when to use it.
    :return:
    """
    # the function has to be described in this format to be understood by the LLM.
    # Caution: Function description is part of the system message, thus uses up tokens and context.
    functions = [
        {
            "name": "weather_search",
            "description": "Search for weather given an airport code",
            "parameters": {
                "type": "object",
                "properties": { # these are the input parameters
                    "airport_code": {
                        "type": "string",
                        "description": "The airport code to get the weather for"
                    },
                },
                "required": ["airport_code"]
            }
        }
    ]
    # this time, we can input anything
    prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}")
        ]
    )
    # The only new thing is that we use "bind(functions=functions)". Again, the LLM is not calling the function,
    # but only tells us when to call
    model = ChatOpenAI(temperature=0).bind(functions=functions)
    runnable = prompt | model
    print(runnable.invoke({"input": "what is the weather in sf"}))

def fallbacks():
    """
    A fallback happens when a chain fails. In this case, we can invoke a fallback chain to repeat the process.
    :return:
    """
    challenge = "write three poems in a json blob, where each poem is a json blob of a title, author, and first line"
    model = ChatOpenAI(temperature=0)
    # We define the first chain, which might use any steps required or even different models.
    s_chain = model | json.loads
    # We define a fallback chain
    chain = model | StrOutputParser() | json.loads
    # We set the fallback chain to be executed in case of failure. We can provide an array of fallbacks,
    # which will be executed one by one if the previous one fails
    final_chain = s_chain.with_fallbacks([chain])
    print(final_chain.invoke(challenge))

if __name__ == '__main__':
    mode = 3
    if mode == 1:
        print("Simple Chain" + '*'*25)
        simple_chain()
    if mode == 2:
        print("Complex Chain" + '*'*25)
        complex_chain()
    if mode == 3:
        print("Function Binding" + '*'*25)
        function_binding()