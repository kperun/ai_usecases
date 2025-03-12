"""
This example shows how we can use function calling together with an LLM to perform tagging (e.g., language of text,
its tone etc.) as well as extraction (e.g., which papers referenced).
"""
import os
import openai

from dotenv import load_dotenv, find_dotenv
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser, JsonKeyOutputFunctionsParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnableLambda

_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


# We use a pydantic class to describe a function which contains information about a given section.
# Here we can say which tags etc we expect.
class Overview(BaseModel):
    """Overview of a section of text."""
    summary: str = Field(description="Provide a concise summary of the content.")
    language: str = Field(description="Provide the language that the content is written in.")
    keywords: str = Field(description="Provide keywords related to the content.")


# We want to extract information about the papers used
class Paper(BaseModel):
    """Information about papers mentioned."""
    title: str
    author: Optional[str]


class Info(BaseModel):
    """Information to extract"""
    papers: List[Paper]


if __name__ == '__main__':
    # We use a webload to load an article
    loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    documents = loader.load()
    model = ChatOpenAI(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Think carefully, and then tag the text as instructed"),
        ("user", "{input}")
    ])
    mode = 1
    doc = documents[0]
    if mode == 1:
        # Example of tagging
        page_content = doc.page_content[:10000]
        # convert the function to be understandable by the llm
        overview_tagging_function = [
            convert_to_openai_function(Overview)
        ]
        # We force the LLM to use the function
        tagging_model = model.bind(
            functions=overview_tagging_function,
            function_call={"name": "Overview"}
        )
        tagging_chain = prompt | tagging_model | JsonOutputFunctionsParser()
        print(tagging_chain.invoke({"input": page_content}))
    if mode == 2:
        # We want to execute extraction, thus we use the Info class
        paper_extraction_function = [
            convert_to_openai_function(Info)
        ]
        extraction_model = model.bind(
            functions=paper_extraction_function,
            function_call={"name": "Info"}  # force to use the extraction function
        )
        # We use a custom template to make the results better
        template = """A article will be passed to you. Extract from it all papers that are mentioned by this article follow by its author. 
        Do not extract the name of the article itself. If no papers are mentioned that's fine - you don't need to extract any! Just return an empty list.
        Do not make up or guess ANY extra information. Only extract what exactly is in the text."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            ("human", "{input}")
        ])
        # This chain is only using one page
        extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="papers")
        """
        We want to scan the whole document, not only one part. But it wont fit into the context window, thus we have
        to use splits.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=0)


        # To later combine the splits:
        def flatten(matrix):
            flat_list = []
            for row in matrix:
                flat_list += row
            return flat_list


        # This pipeline step shall go over all splits and create one map entry for each split
        prep = RunnableLambda(
            lambda x: [{"input": doc} for doc in text_splitter.split_text(x)]
        )
        # We first extract from the documents inputs for each split, then extract information via the other cahin,
        # and finally combine the results using flatten
        chain = prep | extraction_chain.map() | flatten
        print(chain.invoke(doc.page_content))
