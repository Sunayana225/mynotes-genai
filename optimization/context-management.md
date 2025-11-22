# Context Window Management

## What is a Context Window?

**Definition:**
The maximum number of tokens a model can process at once (input + output combined).

**Token Count:**
- GPT-3.5 Turbo: 4K or 16K tokens
- GPT-4: 8K or 32K tokens
- GPT-4 Turbo: 128K tokens
- Claude 2: 100K tokens
- Claude 3: 200K tokens

**Why It Matters:**
- Determines how much information you can provide
- Affects cost (more tokens = higher cost)
- Impacts response quality
- Limits conversation length

## Token Estimation

**Rough Guidelines:**
- 1 token ≈ 4 characters
- 1 token ≈ 0.75 words
- 100 tokens ≈ 75 words
- 1000 tokens ≈ 750 words

**Example:**
```
"Hello, how are you?" = ~5 tokens
One page of text = ~500 tokens
One book = ~100,000 tokens
```

**Counting Tokens:**
```python
import tiktoken

def count_tokens(text, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

text = "The quick brown fox jumps over the lazy dog"
print(count_tokens(text))  # Output: ~9 tokens
```

## Context Window Strategies

### 1. Truncation

**Simple Truncation:**
- Keep most recent messages
- Drop oldest conversation history
- Preserves context continuity

**Smart Truncation:**
- Keep system prompt (always)
- Keep first user message (context)
- Keep recent N messages
- Drop middle messages

**Example:**
```python
def smart_truncate(messages, max_tokens=4000):
    # Always keep system prompt
    system = messages[0]
    recent = messages[-10:]  # Keep last 10 messages
    
    total_tokens = count_tokens(system) + count_tokens(recent)
    
    if total_tokens > max_tokens:
        # Remove oldest from recent until fits
        while total_tokens > max_tokens:
            recent.pop(0)
            total_tokens = count_tokens(system) + count_tokens(recent)
    
    return [system] + recent
```

### 2. Summarization

**Rolling Summarization:**
- Summarize old conversations
- Keep summary instead of full text
- Append new messages

**Process:**
```
Turn 1-5: Full conversation
Turn 6: Summarize 1-5 → Summary A
Turn 7-10: Full messages + Summary A
Turn 11: Summarize 6-10 → Summary B
Continue...
```
