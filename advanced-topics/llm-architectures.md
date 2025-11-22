# LLM Architectures and Neural Networks

## Transformer Architecture

### The Revolution of Attention Mechanism

**Why Transformers Changed Everything:**
- Previous models (RNNs, LSTMs) processed sequentially
- Transformers process entire sequence in parallel
- Attention mechanism sees all tokens simultaneously
- Dramatically faster training and better results

### Core Components

#### 1. Self-Attention Mechanism

**What it does:**
- Determines which words relate to each other
- Calculates importance scores between all word pairs
- Enables understanding of context and relationships

**Example:**
```
Sentence: "The cat sat on the mat because it was comfortable"
- "it" attends strongly to "mat" (high attention score)
- "sat" attends to "cat" (subject-verb relationship)
- "comfortable" attends to "mat" (describes the mat)
```

**Mathematical Process:**
1. **Query (Q)**: What am I looking for?
2. **Key (K)**: What do I contain?
3. **Value (V)**: What information do I provide?

**Attention Score = softmax(Q × K^T / √d_k) × V**

#### 2. Multi-Head Attention

**Why Multiple Heads?**
- Different heads learn different relationships
- Head 1: Syntactic relationships (grammar)
- Head 2: Semantic relationships (meaning)
- Head 3: Long-range dependencies
- Head 4-8: Other patterns

**Typical Configuration:**
- GPT-3: 96 attention heads
- BERT: 12-16 attention heads
- Each head processes information independently
- Results concatenated and projected

#### 3. Feed-Forward Networks

**Structure:**
```
Input → Linear Layer → ReLU → Linear Layer → Output
```

**Purpose:**
- Process attention output
- Add non-linearity
- Transform representations
- Typically 4x larger than hidden size
