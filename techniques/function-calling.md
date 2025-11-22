# Function Calling and Tool Use

## What is Function Calling?

**Definition:**
Ability of LLMs to generate structured data that calls external functions/APIs based on natural language input.

**Evolution:**
- GPT-3: No native function calling
- GPT-3.5/4: Built-in function calling
- Claude 3: Tool use feature
- Gemini: Function calling support

## How It Works

### Basic Flow

```
1. User: "What's the weather in Paris?"
2. LLM decides: Need to call get_weather function
3. LLM returns: {
     "function": "get_weather",
     "arguments": {"city": "Paris", "country": "France"}
   }
4. Application: Calls actual weather API
5. Application: Returns result to LLM
6. LLM: "The weather in Paris is sunny, 22°C"
```

### Function Definition

**OpenAI Format:**
```python
functions = [
    {
        "name": "get_weather",
        "description": "Get current weather in a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }
    }
]
```

**Example Implementation:**
```python
import openai

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "What's the weather in Tokyo?"}
    ],
    functions=functions,
    function_call="auto"
)

# Check if function was called
if response.choices[0].message.get("function_call"):
    function_name = response.choices[0].message["function_call"]["name"]
    arguments = json.loads(response.choices[0].message["function_call"]["arguments"])
    
    # Execute the function
    result = get_weather(arguments["location"])
    
    # Send result back
    second_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "What's the weather in Tokyo?"},
            response.choices[0].message,
            {"role": "function", "name": function_name, "content": str(result)}
        ]
    )
```

## Common Use Cases

### 1. Database Queries

**Function:**
```python
def query_database(table, filters):
    """Query database with filters"""
    sql = f"SELECT * FROM {table} WHERE {filters}"
    return execute_query(sql)
```

**User Query:**
"Show me all orders from last week"

**LLM Output:**
```json
{
  "function": "query_database",
  "arguments": {
    "table": "orders",
    "filters": "date >= DATE_SUB(NOW(), INTERVAL 7 DAY)"
  }
}
```
