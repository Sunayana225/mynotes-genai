# OCI Generative AI Service

OCI Generative AI Service is a fully managed service that provides a set of customizable LLMs available via single API to build generative AI applications.

## Key Features

- **Choice of models**: High performing pretrained models from Meta and Cohere
- **Flexible fine tuning**: Create custom models by fine-tuning foundational models with your own dataset
- **Dedicated AI clusters**: GPU based compute resources that host fine tuning and inference workloads

OCI Gen AI is used to build applications that understand and generate human language at a massive scale.

**Use cases**: Chat, text generation, information retrieval, semantic search

**Flow**: Text is given as input → OCI GenAI Service → Answer as text output

## Pretrained Foundational Models

### Chat Models
Ask questions and get conversational responses:

- **Command R Plus model by Cohere** - More powerful, can handle larger number of requests, is expensive and has more tokens
- **Command R 16k model by Cohere**
- **Llama3-70b-instruct**

### Embedding Models
Convert text to vector embeddings for semantic search:

- **english-v3.0**
- **embed multilingual-v3.0**

### Fine-tuning
Optimizing a pretrained model trained on smaller and specific data which are domain specific.

**Benefits**:
- Improves model performance
- Improves model efficiency

**Use when**:
- A pretrained model doesn't work on your task well
- You want to teach it something new

T-fine-tuning enables fast and efficient customizations:

```
Pretrained Model → Custom Model
                     ↑
                Custom Data
```

## Dedicated AI Clusters

GPU based compute resources that host the customers' fine-tuning and inference workloads.

## Large Language Models (LLMs) - Deep Dive

### What are LLMs?
Large Language Models are neural networks trained on vast amounts of text data to understand and generate human-like text. They use transformer architecture with billions of parameters.

**Key Characteristics:**
- **Scale**: Billions to trillions of parameters
- **Training**: Massive datasets (books, websites, code)
- **Capabilities**: Text generation, translation, summarization, Q&A
- **Context window**: Number of tokens the model can process at once

### Popular LLM Families

#### 1. GPT Series (OpenAI)
- **GPT-3.5**: 175B parameters, efficient for most tasks
- **GPT-4**: Multimodal (text + images), superior reasoning
- **GPT-4 Turbo**: Larger context window (128k tokens)
- Use cases: Chatbots, content creation, code generation

#### 2. Claude (Anthropic)
- **Claude 2**: 100k token context window
- **Claude 3**: Multiple variants (Haiku, Sonnet, Opus)
- Focus on safety and helpfulness
- Excellent for long documents

#### 3. LLaMA (Meta)
- Open-source foundation model
- LLaMA 2: Available for commercial use
- LLaMA 3: Improved performance
- Community fine-tuned variants

#### 4. Gemini (Google)
- Multimodal from the ground up
- Ultra, Pro, Nano variants
- Integration with Google services

### LLM Parameters Explained

**Temperature (0.0 - 2.0)**
- Controls randomness in output
- Low (0.0-0.3): Focused, deterministic
- Medium (0.5-0.7): Balanced creativity
- High (0.8-2.0): Creative, unpredictable

**Top-P / Nucleus Sampling (0.0 - 1.0)**
- Alternative to temperature
- Considers tokens with cumulative probability
- 0.9 means consider top 90% probable tokens

**Max Tokens**
- Maximum length of generated response
- Important for cost and performance
- Typical: 500-4000 tokens

**Frequency Penalty (-2.0 to 2.0)**
- Reduces repetition
- Positive values discourage repeated tokens
- Useful for creative writing

**Presence Penalty (-2.0 to 2.0)**
- Encourages topic diversity
- Positive values introduce new topics
- Negative values stay on topic

### Prompt Engineering Best Practices

**1. Be Specific and Clear**
```
❌ Bad: "Tell me about dogs"
✅ Good: "List 5 key differences between Golden Retrievers and Labrador Retrievers"
```

**2. Provide Context**
```
❌ Bad: "Write a summary"
✅ Good: "Write a 3-sentence summary of this research paper for a high school student"
```

**3. Use Examples (Few-Shot Learning)**
```
Example 1: Input -> Output
Example 2: Input -> Output
Now your input: [user input]
```

**4. Chain of Thought**
```
"Let's solve this step by step:
1. First, identify...
2. Then, calculate...
3. Finally, conclude..."
```

**5. System Prompts**
- Define the AI's role and behavior
- Set tone and constraints
- Example: "You are a helpful coding assistant specializing in Python"

### RAG (Retrieval-Augmented Generation)

**What is RAG?**
Combines LLMs with external knowledge retrieval to provide accurate, up-to-date information.

**How RAG Works:**
```
User Query → Embed Query → Search Vector DB → Retrieve Relevant Docs → 
Include in Prompt → LLM Generates Answer → Return to User
```

**RAG Architecture Components:**

1. **Document Processing**
   - Split documents into chunks (200-1000 tokens)
   - Generate embeddings for each chunk
   - Store in vector database

2. **Query Processing**
   - Convert user question to embedding
   - Find top-K similar chunks
   - Retrieve original text

3. **Context Assembly**
   - Combine retrieved chunks
   - Add to LLM prompt
   - Generate answer based on context

4. **Response Generation**
   - LLM reads context
   - Generates accurate answer
   - Cites sources if needed

**RAG Benefits:**
- ✅ Reduces hallucinations
- ✅ Provides up-to-date information
- ✅ Domain-specific knowledge
- ✅ Source attribution
- ✅ Cost-effective vs fine-tuning

**RAG vs Fine-Tuning:**

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| Cost | Lower | Higher |
| Update Speed | Instant | Requires retraining |
| Flexibility | High | Medium |
| Accuracy | Good | Better |
| Best For | Knowledge retrieval | Specific tasks |

### Advanced RAG Techniques

**1. Hybrid Search**
- Combines vector search + keyword search
- Better recall and precision
- Handles edge cases

**2. Reranking**
- Initial retrieval gets 20-50 chunks
- Reranker model scores relevance
- Returns top 3-5 to LLM

**3. Query Expansion**
- Generate multiple versions of query
- Retrieve for each version
- Combine results

**4. Contextual Compression**
- Retrieved chunks may be too long
- Extract only relevant sentences
- Reduces token usage

**5. Self-Querying**
- LLM generates structured query
- Includes filters (date, category)
- More precise retrieval

### Token Optimization Strategies

**Why Token Efficiency Matters:**
- Reduces API costs
- Faster response times
- Allows longer conversations
- Better user experience

**Optimization Techniques:**

1. **Prompt Caching**
   - Cache system prompts
   - Reuse for multiple requests
   - Reduces redundant processing

2. **Semantic Compression**
   - Summarize long context
   - Extract key information
   - Maintain meaning with fewer tokens

3. **Streaming Responses**
   - Return tokens as generated
   - Better perceived performance
   - User sees results faster

4. **Batch Processing**
   - Process multiple requests together
   - Reduce overhead
   - Cost savings

### LLM Security and Safety

**Common Risks:**
- Prompt injection attacks
- Data leakage
- Bias and fairness issues
- Harmful content generation
- Privacy concerns

**Mitigation Strategies:**

1. **Input Validation**
   - Sanitize user inputs
   - Detect injection attempts
   - Rate limiting

2. **Output Filtering**
   - Content moderation
   - PII detection
   - Toxicity screening

3. **Access Controls**
   - Authentication and authorization
   - API key management
   - Usage monitoring

4. **Audit Logging**
   - Track all interactions
   - Monitor for abuse
   - Compliance requirements

### LLM Evaluation Metrics

**Quantitative Metrics:**
- **Perplexity**: How well model predicts text (lower is better)
- **BLEU Score**: Translation quality
- **ROUGE Score**: Summarization quality
- **Accuracy**: Task-specific correctness

**Qualitative Metrics:**
- Relevance to query
- Factual accuracy
- Coherence and fluency
- Helpfulness
- Safety and bias

**Human Evaluation:**
- A/B testing with users
- Expert review panels
- User satisfaction scores
- Task completion rates

### Cost Optimization for LLMs

**Pricing Models:**
- Pay per token (input + output)
- GPT-4: ~$0.03/1K input, $0.06/1K output
- GPT-3.5: ~$0.001/1K input, $0.002/1K output
- Claude: Similar pricing tiers

**Cost Reduction Strategies:**

1. **Model Selection**
   - Use smaller models when possible
   - GPT-3.5 for simple tasks
   - GPT-4 for complex reasoning

2. **Caching**
   - Cache common responses
   - Deduplicate similar queries
   - Store embeddings

3. **Prompt Engineering**
   - Shorter, more efficient prompts
   - Remove unnecessary context
   - Use prompt templates

4. **Fallback Strategy**
   - Try cheaper model first
   - Escalate to expensive model if needed
   - Hybrid approach

### Future of LLMs

**Emerging Trends:**
- **Multimodal Models**: Text, image, audio, video
- **Longer Context**: Million+ token windows
- **Edge Deployment**: Run locally on devices
- **Specialized Models**: Domain-specific LLMs
- **Efficient Architectures**: Mixture of Experts
- **Personalization**: User-adapted models

**Challenges Ahead:**
- Computational efficiency
- Hallucination reduction
- Bias mitigation
- Interpretability
- Environmental impact

## Vector Databases for GenAI

### Purpose
Store and retrieve high-dimensional embeddings efficiently for semantic search and RAG applications.

### Popular Options

**Pinecone**
- Fully managed cloud service
- Auto-scaling
- Low latency
- $70+/month

**Weaviate**
- Open source + cloud
- GraphQL API
- Hybrid search
- Free tier available

**Qdrant**
- Rust-based, fast
- Easy deployment
- Rich filtering
- Self-hosted or cloud

**Chroma**
- Python-first
- Simple API
- Perfect for prototyping
- Completely free

**Milvus**
- Enterprise-grade
- Highly scalable
- Open source
- Active community

### Key Features to Consider
- Query speed (latency)
- Scalability (millions of vectors)
- Filtering capabilities
- Cost (storage + compute)
- Ease of integration
- Update frequency support

## Building Production GenAI Applications

### Architecture Best Practices

**1. Modular Design**
```
User Interface → API Gateway → LLM Service → Vector DB
                              ↓
                         Embedding Service
```

**2. Observability**
- Log all LLM interactions
- Monitor token usage
- Track latency and errors
- User feedback collection

**3. Caching Strategy**
- Cache embeddings
- Cache common queries
- Cache LLM responses
- Implement TTL appropriately

**4. Error Handling**
- Graceful degradation
- Retry with exponential backoff
- Fallback responses
- User-friendly error messages

**5. Scaling Considerations**
- Async processing for long tasks
- Queue management
- Load balancing
- Database connection pooling

### Deployment Options

**Cloud Providers:**
- **AWS**: Bedrock, SageMaker
- **Azure**: OpenAI Service, ML Studio
- **GCP**: Vertex AI, PaLM API
- **OCI**: Generative AI Service

**Self-Hosted:**
- Run open-source LLMs locally
- Use GPUs (NVIDIA A100, H100)
- Implement model serving (vLLM, TGI)
- Container orchestration (Kubernetes)

### Monitoring and Maintenance

**Key Metrics:**
- Token consumption per user
- Response time (p50, p95, p99)
- Error rates
- User satisfaction scores
- Cost per query

**Continuous Improvement:**
- A/B test prompt variations
- Update knowledge base regularly
- Fine-tune on user feedback
- Monitor for drift and degradation

This comprehensive addition covers advanced GenAI topics that complement your existing notes!



