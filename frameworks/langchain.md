# LangChain Framework

## What is LangChain?

**Definition:**
A framework for developing applications powered by language models, providing modular components and chains for building complex LLM applications.

**Created by:** Harrison Chase (2022)
**Language:** Python & JavaScript/TypeScript

## Core Concepts

### 1. Models

**LLM Wrappers:**
- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude)
- Cohere
- HuggingFace models
- Local models (LLaMA, Mistral)

**Chat Models:**
- Specialized for conversations
- Support system/user/assistant roles
- Message history management

**Embedding Models:**
- OpenAI embeddings
- HuggingFace embeddings
- Cohere embeddings

### 2. Prompts

**Prompt Templates:**
```python
from langchain.prompts import PromptTemplate

template = """
You are a helpful assistant.
Question: {question}
Answer:
"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=template
)

# Usage
prompt.format(question="What is AI?")
```

**Few-Shot Prompt Templates:**
```python
from langchain.prompts import FewShotPromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"}
]

# Creates prompts with examples
```

**Chat Prompt Templates:**
```python
from langchain.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a coding expert"),
    ("user", "{user_input}")
])
```

### 3. Chains

**Simple Chain:**
```python
from langchain.chains import LLMChain

chain = LLMChain(
    llm=model,
    prompt=prompt
)

result = chain.run("What is machine learning?")
```
