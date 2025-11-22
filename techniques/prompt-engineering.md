# Prompt Engineering Techniques

## Advanced Prompting Strategies

### 1. Zero-Shot Prompting

**Definition:**
Ask the model to perform a task without any examples.

**Example:**
```
Classify this review as positive or negative:
"The product broke after two days. Terrible quality."

Output: Negative
```

**When to use:**
- Simple, well-defined tasks
- Model already has domain knowledge
- Need quick results

### 2. Few-Shot Prompting

**Definition:**
Provide examples before asking for the task.

**Example:**
```
Classify sentiment:

Review: "Amazing product! Love it!"
Sentiment: Positive

Review: "Waste of money. Poor quality."
Sentiment: Negative

Review: "The service was fast and helpful."
Sentiment: ?
```

**Best practices:**
- Use 2-5 examples (sweet spot)
- Make examples diverse
- Show edge cases
- Keep consistent format

### 3. Chain-of-Thought (CoT)

**Definition:**
Guide the model to think step-by-step before answering.

**Example:**
```
Question: If John has 5 apples and gives 2 to Mary, then buys 3 more, how many does he have?

Let's solve this step by step:
1. John starts with 5 apples
2. He gives 2 to Mary: 5 - 2 = 3 apples
3. He buys 3 more: 3 + 3 = 6 apples

Answer: John has 6 apples
```

**When to use:**
- Math problems
- Complex reasoning
- Multi-step tasks
- Need to verify logic

### 4. Self-Consistency

**Technique:**
- Generate multiple reasoning paths
- Take majority vote on answers
- Improves accuracy by 10-20%

**Process:**
```
Run same prompt 5 times with temperature > 0
Path 1: Answer A
Path 2: Answer A
Path 3: Answer B
Path 4: Answer A
Path 5: Answer A

Final Answer: A (appears most frequently)
```
