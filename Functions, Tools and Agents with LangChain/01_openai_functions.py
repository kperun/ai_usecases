import os
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file

import json


# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)


# We have to describe the function in a way where OpenAI understands what it provides.
# each function has to be its own entry
functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }
]


def start():
    messages = [
        {
            "role": "user",
            "content": "What's the weather like in Boston?"
        }
    ]
    client = OpenAI(
        api_key=os.environ['OPENAI_API_KEY'],  # This is the default and can be omitted
    )
    response = client.chat.completions.create(
        messages=messages,
        model="gpt-3.5-turbo",
        functions=functions
    )
    # the function call is part of the "function call" of the response
    print(response)
    response_message = response.choices[0].message

    if response_message.function_call is not None:
        # Arguments can be retrieved
        args = json.loads(response_message.function_call.arguments)
        # and finally the function is executed
        function_return = get_current_weather(args)
        messages.append(
            {
                "role": "function",# the role is important to tell the llm that the function was called
                "name": "get_current_weather",
                "content": function_return,
            }
        )
        response = client.chat.completions.create(
            messages=messages,
            model="gpt-3.5-turbo",
            functions=functions
        )
        print(response.choices[0].message.content)
    else:
        print(response.choices[0].message.content)


if __name__ == '__main__':
    print("Execution" + '*' * 40)
    start()