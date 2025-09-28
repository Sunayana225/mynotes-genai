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



