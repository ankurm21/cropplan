Function calling with the Gemini API
Function calling lets you connect models to external tools and APIs. Instead of generating text responses, the model determines when to call specific functions and provides the necessary parameters to execute real-world actions. This allows the model to act as a bridge between natural language and real-world actions and data. Function calling has 3 primary use cases:
Take Actions: Interact with external systems using APIs, such as scheduling appointments, creating invoices, sending emails, or controlling smart home devices.
Augment Knowledge: Access information from external sources like databases, APIs, and knowledge bases.
Extend Capabilities: Use external tools to perform computations and extend the limitations of the model, such as using a calculator or creating charts.
You can browse examples of these use cases below:
Schedule Meeting 
This example shows how to define a function that schedules a meeting with attendees at a specific time, allowing the model to parse user requests and return structured arguments to trigger actions in external systems.
Python
JavaScript
REST
Java
Go
code
sample.py
from google import genai

schedule_meeting_function = {
    "type": "function",
    "name": "schedule_meeting",
    "description": "Schedules a meeting with specified attendees at a given time and date.",
    "parameters": {
        "type": "object",
        "properties": {
            "attendees": {"type": "array", "items": {"type": "string"}},
            "date": {"type": "string", "description": "Date (e.g., '2024-07-29')"},
            "time": {"type": "string", "description": "Time (e.g., '15:00')"},
            "topic": {"type": "string", "description": "The meeting topic."},
        },
        "required": ["attendees", "date", "time", "topic"],
    },
}

client = genai.Client()

interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input="Schedule a meeting with Bob and Alice for 03/14/2025 at 10:00 AM about Q3 planning.",
    tools=[{"type": "function", **schedule_meeting_function}],
)

for step in interaction.steps:
    if step.type == "function_call":
        print(f"Function to call: {step.name}")
        print(f"Arguments: {step.arguments}")
Get Weather 
This example shows how to define a function that retrieves temperature data for a location, enabling the model to call external APIs to answer queries requiring real-time or external information.
Python
JavaScript
REST
Java
Go
code
sample.py
from google import genai

weather_function = {
    "type": "function",
    "name": "get_current_temperature",
    "description": "Gets the current temperature for a given location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city name, e.g. San Francisco",
            },
        },
        "required": ["location"],
    },
}

client = genai.Client()

interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input="What's the temperature in London?",
    tools=[weather_function],
)

for step in interaction.steps:
    if step.type == "function_call":
        print(f"Function to call: {step.name}")
        print(f"Arguments: {step.arguments}")
Create Chart 
This example shows how to define a function that generates a bar chart from structured data, demonstrating how the model can use external tools to perform computations or create visual assets:
Python
JavaScript
REST
Java
Go
code
sample.py
from google import genai

create_chart_function = {
    "type": "function",
    "name": "create_bar_chart",
    "description": "Creates a bar chart given a title, labels, and values.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "The title for the chart."},
            "labels": {"type": "array", "items": {"type": "string"}},
            "values": {"type": "array", "items": {"type": "number"}},
        },
        "required": ["title", "labels", "values"],
    },
}

client = genai.Client()

interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input="Create a bar chart titled 'Quarterly Sales' with Q1: 50000, Q2: 75000, Q3: 60000.",
    tools=[create_chart_function],
)

for step in interaction.steps:
    if step.type == "function_call":
        print(f"Function to call: {step.name}")
        print(f"Arguments: {step.arguments}")
How function calling works 
undefined
Function calling involves a structured interaction between your application, the model, and external functions:
Define Function Declaration: Define the function's name, parameters, and purpose to the model.
Call LLM with function declarations: Send user prompt along with the function declaration(s) to the model.
Execute Function Code (Your Responsibility): The model doesn't execute the function itself. Extract the name and args and execute in your application.
Create User friendly response: Send the result back to the model for a final, user-friendly response.
This process can be repeated over multiple turns. The model supports calling multiple functions in a single turn (parallel function calling) and in sequence (compositional function calling).
Step 1: Define a function declaration 
Python
JavaScript
Java
Go
code
sample.py
set_light_values_declaration = {
    "type": "function",
    "name": "set_light_values",
    "description": "Sets the brightness and color temperature of a light.",
    "parameters": {
        "type": "object",
        "properties": {
            "brightness": {
                "type": "integer",
                "description": "Light level from 0 to 100",
            },
            "color_temp": {
                "type": "string",
                "enum": ["daylight", "cool", "warm"],
                "description": "Color temperature",
            },
        },
        "required": ["brightness", "color_temp"],
    },
}

def set_light_values(brightness: int, color_temp: str) -> dict:
    """Set the brightness and color temperature of a room light."""
    return {"brightness": brightness, "colorTemperature": color_temp}
Step 2: Call the model with function declarations 
Python
JavaScript
Java
Go
code
sample.py
from google import genai

client = genai.Client()

interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input="Turn the lights down to a romantic level",
    tools=[set_light_values_declaration],
)

fc_step = next(s for s in interaction.steps if s.type == "function_call")
print(fc_step)
The model returns a function_call step with type, name, and arguments:
code
Python
type='function_call'
name='set_light_values'
arguments={'color_temp': 'warm', 'brightness': 25}
Step 3: Execute the function 
Python
JavaScript
Java
Go
code
sample.py
fc_step = next(s for s in interaction.steps if s.type == "function_call")

if fc_step.name == "set_light_values":
    result = set_light_values(**fc_step.arguments)
    print(f"Function execution result: {result}")
Step 4: Send result back to model 
Python
JavaScript
Java
Go
code
sample.py
final_interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input=[
        {
            "type": "function_result",
            "name": fc_step.name,
            "call_id": fc_step.id,
            "result": [{"type": "text", "text": json.dumps(result)}],
        }
    ],
    tools=[set_light_values_declaration],
    previous_interaction_id=interaction.id,
)

print(final_interaction.output_text)
Stateless function calling 
You can also use function calling in stateless mode by managing the conversation history on the client side and setting store=false.
In stateless mode, you must pass the full history of the conversation in the input field of each subsequent request. This history must include:
The initial user_input step.
All model-generated steps returned in Turn 1 (including thought and function_call steps) exactly as received.
The function_result step containing the output of your executed function.
Python
JavaScript
REST
Java
Go
code
sample.py
from google import genai
import json

client = genai.Client()

history = [
    {
        "type": "user_input",
        "content": [{"type": "text", "text": "Turn the lights down to a romantic level"}]
    }
]

interaction = client.interactions.create(
    model="gemini-3.8-flash",
    store=False,
    input=history,
    tools=[set_light_values_declaration],
)

for step in interaction.steps:
    history.append(step.model_dump())

fc_step = next(s for s in interaction.steps if s.type == "function_call")
if fc_step.name == "set_light_values":
    result = set_light_values(**fc_step.arguments)

history.append({
    "type": "function_result",
    "name": fc_step.name,
    "call_id": fc_step.id,
    "result": [{"type": "text", "text": json.dumps(result)}],
})

final_interaction = client.interactions.create(
    model="gemini-3.8-flash",
    store=False,
    input=history,
    tools=[set_light_values_declaration],
)

print(final_interaction.output_text)
Function declarations 
A function declaration is passed as a tool and includes:
type (string): Must be "function" for custom functions.
name (string): Unique function name (use underscores or camelCase).
description (string): Clear explanation of the function's purpose.
parameters (object): Input parameters the function expects.
type (string): Overall data type, such as object.
properties (object): Individual parameters with type and description.
required (array): Mandatory parameter names.
