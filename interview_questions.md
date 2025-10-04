# 200 LLM Interview Questions & Answers

## Table of Contents
1. [Basic Level (1-50)](#basic-level-1-50)
2. [Intermediate Level (51-120)](#intermediate-level-51-120)
3. [Advanced Level (121-180)](#advanced-level-121-180)
4. [Expert Level (181-200)](#expert-level-181-200)
5. [Quick Reference](#quick-reference)

---

## Basic Level (1-50)

### 1. What is a Large Language Model (LLM)?
**Answer:** A Large Language Model is a neural network-based AI system trained on massive amounts of text data to understand, generate, and manipulate human language. The "large" refers to the number of parameters (typically billions) and the extensive training data.

### 2. What are parameters in an LLM?
**Answer:** Parameters are the learnable weights and biases in the neural network that are adjusted during training. They determine how the model processes input and generates output.

### 3. What is the transformer architecture?
**Answer:** The transformer is a neural network architecture introduced in the "Attention Is All You Need" paper (2017) that uses self-attention mechanisms to process sequential data, making it highly parallelizable and effective for language tasks.

### 4. What is tokenization?
**Answer:** Tokenization is the process of breaking down text into smaller units (tokens) that the model can process. Tokens can be words, subwords, or characters.

### 5. What are embeddings?
**Answer:** Embeddings are numerical representations of tokens in a high-dimensional space where similar words have similar vector representations, capturing semantic relationships.

### 6. What is fine-tuning?
**Answer:** Fine-tuning is the process of taking a pre-trained model and training it further on a specific dataset to adapt it to a particular task or domain.

### 7. What is prompt engineering?
**Answer:** Prompt engineering is the practice of designing and optimizing input prompts to get desired outputs from language models.

### 8. What is zero-shot learning?
**Answer:** Zero-shot learning is when a model performs a task without any specific training examples for that task, relying only on its general knowledge.

### 9. What is few-shot learning?
**Answer:** Few-shot learning provides the model with a small number of examples in the prompt to demonstrate the desired task before asking it to perform on new inputs.

### 10. What is temperature in LLM sampling?
**Answer:** Temperature is a hyperparameter that controls the randomness of predictions. Lower temperature (0.1-0.5) makes outputs more deterministic, while higher temperature (0.7-1.0) makes them more creative.

### 11. What is top-p (nucleus) sampling?
**Answer:** Top-p sampling selects from the smallest set of tokens whose cumulative probability exceeds p, allowing dynamic vocabulary size while maintaining quality.

### 12. What is top-k sampling?
**Answer:** Top-k sampling restricts sampling to the k most likely tokens at each step, preventing low-probability tokens from being selected.

### 13. What is beam search?
**Answer:** Beam search is a decoding algorithm that keeps multiple candidate sequences (beams) at each step and selects the most likely overall sequence.

### 14. What are encoder models used for?
**Answer:** Encoder models (like BERT) are designed for understanding tasks - classification, named entity recognition, sentiment analysis, and feature extraction.

### 15. What are decoder models used for?
**Answer:** Decoder models (like GPT) are designed for generative tasks - text completion, story writing, code generation, and conversational AI.

### 16. What is masked language modeling?
**Answer:** Masked language modeling is a training objective where random tokens in input text are masked, and the model learns to predict them based on context.

### 17. What is causal language modeling?
**Answer:** Causal language modeling trains models to predict the next token given previous tokens, making it suitable for text generation.

### 18. What is the difference between training and inference?
**Answer:** Training is the process of learning model parameters from data, while inference is using the trained model to make predictions on new data.

### 19. What is a loss function?
**Answer:** A loss function measures how well the model's predictions match the actual targets, guiding the optimization process during training.

### 20. What is gradient descent?
**Answer:** Gradient descent is an optimization algorithm that minimizes the loss function by iteratively adjusting parameters in the direction of steepest descent.

### 21. What is overfitting?
**Answer:** Overfitting occurs when a model learns the training data too well, including noise and outliers, and performs poorly on unseen data.

### 22. What is regularization?
**Answer:** Regularization techniques prevent overfitting by adding constraints to the model, such as weight decay or dropout.

### 23. What is transfer learning?
**Answer:** Transfer learning involves taking knowledge learned from one task and applying it to a different but related task.

### 24. What is self-attention?
**Answer:** Self-attention allows each position in a sequence to attend to all other positions, capturing long-range dependencies and relationships.

### 25. What are query, key, and value in attention?
**Answer:** In attention mechanisms, queries represent what we're looking for, keys represent what we can offer, and values represent the actual content we want to extract.

### 26. What is multi-head attention?
**Answer:** Multi-head attention runs multiple attention mechanisms in parallel, allowing the model to focus on different aspects of the input simultaneously.

### 27. What is positional encoding?
**Answer:** Positional encoding adds information about the position of tokens in the sequence to the input embeddings, since transformers don't inherently understand order.

### 28. What is layer normalization?
**Answer:** Layer normalization stabilizes training by normalizing activations across features for each data point.

### 29. What is a vocabulary in LLMs?
**Answer:** The vocabulary is the set of all possible tokens the model can recognize and generate.

### 30. What are special tokens?
**Answer:** Special tokens like [CLS], [SEP], [BOS], [EOS] are used for specific purposes like classification, separation, beginning/end of sequences.

### 31. What is perplexity?
**Answer:** Perplexity measures how well a probability model predicts a sample, with lower values indicating better performance.

### 32. What is BLEU score?
**Answer:** BLEU (Bilingual Evaluation Understudy) measures the quality of machine-translated text by comparing it to human references.

### 33. What is ROUGE score?
**Answer:** ROUGE (Recall-Oriented Understudy for Gisting Evaluation) evaluates text summarization by measuring overlap with reference summaries.

### 34. What is BERTScore?
**Answer:** BERTScore uses contextual embeddings to evaluate text similarity, providing more nuanced evaluation than BLEU/ROUGE.

### 35. What is the difference between training, validation, and test sets?
**Answer:** Training set is used to learn parameters, validation set for hyperparameter tuning, and test set for final evaluation.

### 36. What is data augmentation?
**Answer:** Data augmentation artificially increases training data size through techniques like paraphrasing, back-translation, or noise injection.

### 37. What is cross-entropy loss?
**Answer:** Cross-entropy loss measures the difference between predicted and actual probability distributions, commonly used for classification.

### 38. What is teacher forcing?
**Answer:** Teacher forcing uses the ground truth previous token as input during training instead of the model's own predictions.

### 39. What is curriculum learning?
**Answer:** Curriculum learning trains models on progressively difficult examples, from simple to complex.

### 40. What is knowledge distillation?
**Answer:** Knowledge distillation trains a smaller student model to mimic a larger teacher model, preserving performance while reducing size.

### 41. What is model parallelism?
**Answer:** Model parallelism distributes different parts of a model across multiple devices to handle large models that don't fit on single devices.

### 42. What is data parallelism?
**Answer:** Data parallelism replicates the model across multiple devices and processes different data batches in parallel.

### 43. What is mixed precision training?
**Answer:** Mixed precision training uses both 16-bit and 32-bit floating-point types to speed up training and reduce memory usage.

### 44. What is gradient checkpointing?
**Answer:** Gradient checkpointing reduces memory usage by recomputing activations during backward pass instead of storing them.

### 45. What is early stopping?
**Answer:** Early stopping halts training when validation performance stops improving to prevent overfitting.

### 46. What is learning rate scheduling?
**Answer:** Learning rate scheduling adjusts the learning rate during training to improve convergence and performance.

### 47. What is weight decay?
**Answer:** Weight decay is a regularization technique that penalizes large weights by adding their magnitude to the loss function.

### 48. What is dropout?
**Answer:** Dropout randomly sets a fraction of activations to zero during training to prevent overfitting.

### 49. What is batch normalization?
**Answer:** Batch normalization normalizes activations across batches to stabilize and accelerate training.

### 50. What is a transformer block?
**Answer:** A transformer block typically consists of multi-head attention, feed-forward networks, residual connections, and layer normalization.

---

## Intermediate Level (51-120)

### 51. Explain the transformer architecture in detail.
**Answer:** The transformer consists of:

- **Encoder:** Processes input sequence with self-attention and feed-forward layers
- **Decoder:** Generates output with masked self-attention, encoder-decoder attention, and feed-forward layers
- **Multi-head Attention:** Multiple attention heads in parallel
- **Positional Encoding:** Adds sequence order information
- **Residual Connections & Layer Norm:** Stabilizes training

### 52. How does self-attention work mathematically?
**Answer:** Self-attention computes:

```
Attention(Q, K, V) = softmax(QK^T/√d_k)V
```
Where Q, K, V are queries, keys, and values matrices, and d_k is the dimension of keys.

### 53. What is the difference between BERT and GPT architectures?
**Answer:**

- **BERT:** Encoder-only, bidirectional context, good for understanding tasks
- **GPT:** Decoder-only, unidirectional context, good for generation tasks
- **T5:** Encoder-decoder, good for text-to-text tasks

### 54. How does positional encoding work in transformers?
**Answer:** Uses sine and cosine functions of different frequencies:

```
PE(pos, 2i) = sin(pos/10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
```

### 55. What is the purpose of layer normalization in transformers?
**Answer:** Layer normalization stabilizes training by normalizing activations across features for each data point, reducing covariate shift and improving convergence.

### 56. Explain the feed-forward network in transformer blocks.
**Answer:** The FFN consists of two linear transformations with a ReLU activation in between:

```
FFN(x) = max(0, xW1 + b1)W2 + b2
```
It processes each position independently and identically.

### 57. What are residual connections and why are they important?
**Answer:** Residual connections add the input of a layer to its output (x + F(x)), helping with gradient flow and enabling training of very deep networks.

### 58. How does multi-head attention improve performance?
**Answer:** Multi-head attention allows the model to jointly attend to information from different representation subspaces, capturing different types of relationships.

### 59. What is the difference between full attention and causal attention?
**Answer:** Full attention allows each token to attend to all other tokens (bidirectional), while causal attention restricts tokens to attend only to previous tokens (unidirectional).

### 60. How do you handle variable-length sequences in transformers?
**Answer:** Through padding/truncation to fixed lengths, and using attention masks to ignore padding tokens during computation.

61. What is teacher forcing in training?
Answer: Teacher forcing uses the ground truth as input during training rather than the model's own predictions, accelerating convergence but potentially causing exposure bias.

62. What is exposure bias?
Answer: Exposure bias occurs when models are trained with teacher forcing but tested with their own predictions, leading to error accumulation during inference.

63. How does beam search differ from greedy decoding?
Answer: Greedy decoding always picks the most likely next token, while beam search maintains multiple hypotheses and chooses the overall best sequence.

64. What are the trade-offs of different decoding strategies?
Answer:

Greedy: Fast but low quality

Beam Search: Better quality but more compute

Sampling: More diverse but less coherent

Nucleus: Balanced quality and diversity

65. How do you evaluate text generation quality?
Answer: Using automated metrics (BLEU, ROUGE, METEOR), human evaluation, task-specific metrics, and perplexity.

66. What is the difference between intrinsic and extrinsic evaluation?
Answer: Intrinsic evaluation measures linguistic quality directly, while extrinsic evaluation measures performance on downstream tasks.

67. How does fine-tuning differ from feature extraction?
Answer: Fine-tuning updates all model parameters, while feature extraction uses pre-trained features with a new classifier on top.

68. What is catastrophic forgetting?
Answer: Catastrophic forgetting occurs when fine-tuning on new tasks causes the model to forget previously learned knowledge.

69. How do you prevent catastrophic forgetting?
Answer: Through elastic weight consolidation, progressive networks, replay buffers, and multi-task learning.

70. What is domain adaptation?
Answer: Domain adaptation adjusts models to perform well on data from domains different from their training data.

71. How does prompt engineering work?
Answer: Prompt engineering designs inputs to elicit desired behaviors through techniques like few-shot learning, chain-of-thought, and specific formatting.

72. What is chain-of-thought prompting?
Answer: Chain-of-thought prompting encourages models to generate intermediate reasoning steps before producing final answers.

73. What is least-to-most prompting?
Answer: Least-to-most prompting breaks complex problems into simpler subproblems and solves them sequentially.

74. What is step-back prompting?
Answer: Step-back prompting asks models to first identify high-level concepts before solving specific problems.

75. How do you handle long-context in LLMs?
Answer: Through positional encoding extensions (RoPE, ALiBi), hierarchical processing, retrieval augmentation, and memory mechanisms.

76. What is RAG (Retrieval-Augmented Generation)?
Answer: RAG combines retrieval from external knowledge sources with generation to produce more accurate and grounded responses.

77. How does RAG reduce hallucinations?
Answer: RAG grounds generation in retrieved evidence, providing factual basis and reducing fabrication.

78. What are the components of a RAG system?
Answer: Retriever (dense/sparse), vector database, generator, and reranker components.

79. What is the difference between dense and sparse retrieval?
Answer: Dense retrieval uses embeddings and similarity search, while sparse retrieval uses term matching like BM25.

80. How do you evaluate RAG systems?
Answer: Using retrieval metrics (recall, MRR), generation metrics (faithfulness, answer relevance), and end-to-end task performance.

81. What is self-consistency in decoding?
Answer: Self-consistency generates multiple reasoning paths and selects the most consistent answer through voting.

82. How does reinforcement learning help LLM training?
Answer: RL aligns model outputs with human preferences through reward modeling and policy optimization.

83. What is PPO in RLHF?
Answer: Proximal Policy Optimization is a reinforcement learning algorithm used to optimize language models while preventing large policy updates.

84. How does DPO differ from RLHF?
Answer: DPO directly optimizes preference probabilities without explicit reward modeling or reinforcement learning.

85. What are the advantages of LoRA?
Answer: LoRA reduces trainable parameters, enables efficient fine-tuning, and facilitates model merging and sharing.

86. How does quantization affect model performance?
Answer: Quantization typically causes minor performance degradation but significantly reduces memory and computational requirements.

87. What is 4-bit quantization?
Answer: 4-bit quantization represents weights with 4 bits instead of 32 bits, using techniques like GPTQ or QLoRA.

88. How does knowledge distillation work?
Answer: The student model learns to match the teacher's outputs through distillation loss while being trained on the original task.

89. What is speculative decoding?
Answer: Speculative decoding uses a small draft model to generate tokens quickly and a larger verification model to correct them, speeding up inference.

90. How do you optimize LLM inference latency?
Answer: Through model quantization, kernel optimization, batching, caching, and hardware acceleration.

91. What is KV caching?
Answer: KV caching stores key-value pairs from previous tokens to avoid recomputation during autoregressive generation.

92. How does flash attention work?
Answer: Flash attention optimizes attention computation through tiling and recomputation to reduce memory usage and improve speed.

93. What are the challenges of serving LLMs?
Answer: High memory requirements, latency constraints, throughput optimization, and cost management.

94. How do you handle multi-turn conversations?
Answer: By maintaining conversation history, using special tokens for turns, and implementing context window management.

95. What is prompt injection?
Answer: Prompt injection manipulates model inputs to bypass safety measures or extract sensitive information.

96. How do you prevent prompt injection?
Answer: Through input sanitization, prompt separation, model safety training, and output filtering.

97. What is jailbreaking?
Answer: Jailbreaking uses creative prompts to circumvent model safety controls and restrictions.

98. How do you evaluate model safety?
Answer: Using red teaming, safety benchmarks, adversarial testing, and human evaluation.

99. What is model alignment?
Answer: Model alignment ensures model behavior matches human values and intentions through techniques like RLHF and constitutional AI.

100. What are the ethical considerations in LLM deployment?
Answer: Bias mitigation, fairness, transparency, privacy, and responsible use considerations.

101. How do you detect and mitigate bias in LLMs?
Answer: Through bias auditing, debiasing techniques, diverse training data, and fairness constraints.

102. What is data contamination?
Answer: Data contamination occurs when test data appears in training data, leading to inflated performance metrics.

103. How do you handle out-of-distribution inputs?
Answer: Through confidence calibration, outlier detection, and fallback mechanisms.

104. What is calibration in LLMs?
Answer: Calibration ensures model confidence scores accurately reflect true probabilities.

### 105. How do you measure model uncertainty?
**Answer:** Through ensemble methods, Monte Carlo dropout, and predictive entropy.

---

## Quick Reference

### Key Concepts Summary

| Category | Key Terms |
|----------|-----------|
| **Architecture** | Transformer, Self-attention, Multi-head attention, Encoder-Decoder |
| **Training** | Fine-tuning, Pre-training, RLHF, Knowledge distillation |
| **Sampling** | Temperature, Top-p, Top-k, Beam search |
| **Evaluation** | Perplexity, BLEU, ROUGE, BERTScore |
| **Optimization** | Gradient descent, Learning rate scheduling, Mixed precision |
| **Regularization** | Dropout, Weight decay, Early stopping |

### Model Types Quick Reference

| Model Type | Architecture | Best For | Examples |
|------------|-------------|----------|----------|
| **Encoder-only** | BERT-style | Understanding tasks | BERT, RoBERTa, DeBERTa |
| **Decoder-only** | GPT-style | Generation tasks | GPT-3/4, PaLM, LLaMA |
| **Encoder-Decoder** | T5-style | Text-to-text tasks | T5, BART, UL2 |

### Attention Mechanisms

```
Self-Attention: Attention(Q, K, V) = softmax(QK^T/√d_k)V
Multi-Head: MultiHead(Q,K,V) = Concat(head₁,...,headₕ)W^O
Positional Encoding: PE(pos,2i) = sin(pos/10000^(2i/d_model))
```

### Training Strategies

1. **Pre-training:** Large-scale unsupervised learning
2. **Fine-tuning:** Task-specific supervised learning
3. **RLHF:** Reinforcement Learning from Human Feedback
4. **In-context Learning:** Learning from examples in prompt

### Common Interview Topics

#### Technical Deep Dives
- Transformer architecture details
- Attention mechanism mathematics
- Training optimization techniques
- Model parallelism and scaling

#### Practical Applications
- Prompt engineering strategies
- Fine-tuning approaches
- Evaluation methodologies
- Deployment considerations

#### Ethics and Safety
- Bias detection and mitigation
- Model alignment techniques
- Safety evaluation methods
- Responsible AI practices

---

## Study Tips

### For Technical Roles
1. **Understand the math** - Know attention formulas and training objectives
2. **Practice coding** - Implement basic transformer components
3. **Study papers** - Read key papers like "Attention Is All You Need"
4. **Hands-on experience** - Fine-tune models on specific tasks

### For Product/Research Roles
1. **Focus on applications** - Understand use cases and limitations
2. **Learn evaluation** - Know how to measure model performance
3. **Ethics awareness** - Understand bias, safety, and alignment issues
4. **Industry trends** - Stay updated on latest model developments

### Common Pitfalls to Avoid
- Confusing different attention types
- Not understanding the difference between training and inference
- Mixing up encoder vs decoder architectures
- Forgetting about computational complexity and scaling issues

---

*This document contains 105+ essential LLM interview questions covering basic to advanced topics. Review regularly and practice explaining concepts in your own words for best interview preparation.*