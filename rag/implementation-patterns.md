# RAG Implementation Patterns

## Advanced RAG Architectures

### 1. Naive RAG (Basic Implementation)

**Flow:**
```
Query → Embed → Search Vector DB → Retrieve Top-K → 
Add to Prompt → LLM → Response
```

**Pros:**
- Simple to implement
- Fast to set up
- Works for basic use cases

**Cons:**
- May retrieve irrelevant chunks
- No query understanding
- Context may be insufficient
- Can't handle complex queries

### 2. Advanced RAG

**Enhancements:**

**Pre-Retrieval:**
- Query rewriting
- Query expansion
- Hypothetical document embeddings (HyDE)

**Retrieval:**
- Hybrid search (vector + keyword)
- Multiple vector stores
- Metadata filtering
- Semantic ranking

**Post-Retrieval:**
- Reranking retrieved chunks
- Contextual compression
- Chunk fusion
- Citation generation

### 3. Self-RAG

**What it does:**
Model decides when to retrieve and what to trust.

**Process:**
1. Receive query
2. Model decides: "Do I need retrieval?"
3. If yes, retrieve relevant docs
4. Generate answer
5. Self-critique: "Is this factual?"
6. Revise if needed

**Benefits:**
- Reduces unnecessary retrievals
- Better factuality
- Self-correction capability

### 4. Corrective RAG (CRAG)

**What it does:**
Evaluates retrieved documents and corrects course.

**Steps:**
1. Retrieve documents
2. Score relevance of each doc
3. If irrelevant: Expand search or use web search
4. If ambiguous: Combine multiple sources
5. If relevant: Proceed with generation

**Scoring:**
- Correct: High relevance
- Incorrect: Low relevance
- Ambiguous: Mixed signals

### 5. Adaptive RAG

**What it does:**
Routes queries to different strategies based on complexity.

**Routing Logic:**
```
Simple factual query → Direct retrieval
Complex reasoning → Multi-hop retrieval
Calculation needed → Tool calling
Current events → Web search
```
