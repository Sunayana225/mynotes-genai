# Evaluation Metrics for LLMs

## Why Evaluation Matters

**Challenges:**
- Subjective quality (what's "good"?)
- Task diversity
- Context-dependent correctness
- Balancing multiple objectives

## Automatic Metrics

### 1. Perplexity

**What it measures:**
How well the model predicts the next token.

**Formula:**
```
Perplexity = exp(average negative log-likelihood)
```

**Interpretation:**
- Lower is better
- Perplexity of 10 = model is as uncertain as choosing from 10 options
- Good for comparing models on same dataset

**Limitations:**
- Doesn't correlate with human judgment
- Can't measure factuality
- Doesn't capture usefulness

### 2. BLEU Score (Bilingual Evaluation Understudy)

**What it measures:**
Overlap between generated text and reference text.

**Used for:**
- Machine translation
- Text generation quality
- Summarization

**Range:** 0-100 (higher is better)

**Example:**
```
Reference: "The cat sat on the mat"
Generated: "The cat is on the mat"
BLEU Score: ~70
```

**Limitations:**
- Doesn't understand meaning
- Penalizes valid paraphrases
- Focuses on n-gram overlap

### 3. ROUGE Score (Recall-Oriented Understudy for Gisting Evaluation)

**What it measures:**
Recall of n-grams between generated and reference text.

**Variants:**
- **ROUGE-N**: N-gram overlap
- **ROUGE-L**: Longest common subsequence
- **ROUGE-S**: Skip-bigram co-occurrence

**Used for:**
- Summarization evaluation
- Content coverage assessment

**Example:**
```
ROUGE-1: Unigram overlap
ROUGE-2: Bigram overlap
ROUGE-L: Longest matching sequence
```

### 4. BERTScore

**What it measures:**
Semantic similarity using BERT embeddings.

**Advantages:**
- Understands paraphrases
- Captures semantic meaning
- Better correlation with human judgment

**Process:**
1. Embed reference and generated text
2. Compute cosine similarity
3. Match tokens with highest similarity
4. Aggregate scores
