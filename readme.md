# Personal StuMultiple archit- For example, the emb## Encoders

![alt t## Decoders

Models take a sequence of words and output the next word.

**Examples:** GPT-4, Llama, Bloom, Cohere

What these models do is they take a sequence of tokens and emit the next token in the sequence based on the probability of the vocabulary which they compute. A decoder produces only a single token at a time.

We can always invoke a decoder over and over to generate many new tokens as we want. To do that we need to:

1. First feed in a sequence of tokens and invoke the model to produce the next token
2. Then append the generated token to the input sequence and feed it back to the model so that it can produce the second token

It is a very expensive process and typically we would not use a decoder model for embedding. Decoders are being used now for text generation as they have shown tremendous amounts of capability to do so.

## Encoder-Decoder

Encodes a specific sequence of words and uses the encoding to output the next word.1.png)

Models that convert a sequence of words to an embedding vector representation. These vector representations are designed to be consumed by later models to do things like classification or regression.

A lot of their use these days is for semantic search or vector search in databases. Let's say we are wanting to retrieve a document from a corpus:

1. To accomplish this you could encode or synonymously embed each document in a corpus and store them in an index
2. When you get an input snippet you encode that too and check the similarity with the encoded input against the similarity of each document of the corpus and return the most similar one

## Decodersr "king" and "queen" are closer than for "king" and "banana"

All models are built on transformer architecture. Each type of model has different capabilities (embedding or generation). Encoders and decoders have different sizes.

In the realm of language models, size = number of trainable parameters that the model has.

![alt text](image.png)

Decoders are relatively large compared to encoders. When models are too small they have proven to be bad text generators. With a bit of added cleverness we may generate better text.

**Refer to the PDF to have more content on this topic.**

## Encoderslt on encoding and decoding focus on embedding and text generation.

Embedding is a way of representing words, tokens or entire pieces of text as numerical vectors so that the neural network can understand and process them.

### What an Embedding Represents

- Each token (word, subword, or character) is turned into a vector of, say, 768 or 1024 numbers
- The position of the vector in this space captures semantic relationships
- Words with similar meaning end up closer together
- For example, the embeddings for "king" and "queen" are closer than for "king" and "banana"s - Oracle Gen AI Course

**Disclaimer:** These are my personal study notes for the Oracle Gen AI course and are not official course material. They are for educational purposes and personal reference only.

---

## LLM - Large Language Model

A language model is a very probabilistic model of text. Large in large language model (LLM) refers to the number of parameters; there is no agreed upon threshold.

## LLM Architectures

### Encoders and Decoders

Multiple architectures built on encoding and decoding focus on embedding and text generation.  
embedding is a way of representing words,tokens or entire pieces of text a numerical vectors so that the neural network can understand and process them
What an Embedding Represents
Each token (word, subword, or character) is turned into a vector of, say, 768 or 1024 numbers.
The position of the vector in this space captures semantic relationships.
Words with similar meaning end up closer together.
e.g., the embeddings for “king” and “queen” are closer than for “king” and “banana.”
All models are built on transformer architecture 
each type of model has different capabilities (embedding or generation)
 it encoders and decoders have different sizes
in the realm of language model size=no.of trainable parameters that model has 
![alt text](image.png)
decoders are relatively large compare to encoders 
when models are too small they have proven to be bad text generators  
with a bit of added cleverness we may generate better text 
Refer to the pdf to have more content on this topic 
encoders:
![alt text](image-1.png)
models that convert a sequence of words to an embedding vector representation
these vector representations are designed to be consumed by later models to do things like classification or regression 
But a lot of their use these days is for semantic search or vector search in databases 
lets say we are wanting to retrieve a document from a corpus 
to accomplish this you could encode or synonymously embed
each doc in a corpus and store them in an index
When you get an input snippet you encode that too and check the similarity with the encoded input against the similarity of each document of the corpus and return the most similar one 
Decoders :
Models take a sequence of words and output next word 
examples:GPT 4 ,Llama,bloom,cohere
What these models do is they take a sequence of tokens and emit the next token in the sequence based on the probability of the vocabulary which they compute 
a decoder produces only single token at a time 
we can always invoke a decoder over and over to generate many new tokens as we want
To do that we need to first feed in a sequence of tokens and invoke model to produce the next token then append the generated token to generate the input sequence and feed it back to the model so that it can produce the second token 
it is a very expensive process and typically we would not use decoder model for embedding
decoders are being used now for generation text as they have shown tremendous amounts of capability to do so 
encoder-decoder
encodes a specific sequence of words and use the encoding + to output a next word
![alt text](image-2.png)
![alt text](image-3.png)
## Chain of Thought Prompting

If we have a complicated set of tasks, prompt the LLM to emit immediate reasoning steps.

**Least to Most Prompting:** Prompt the LLM to decompose the problem and solve easy first. Basically it was taught to do the easiest task first.

**Step Back:** Prompt the LLM to identify high level concepts pertinent to a specific task.

Prompting is very critical and unreliable sometimes because even a small space could make a great difference.

## Prompt Injection

### Issues with Prompting

Prompt injection is a process to provide the LLM with input that attempts to cause it to ignore instructions, cause harm, or behave differently, or behave contrarily to deployment expectations.

It can be used to extract some information which is not supposed to be exposed like private information.

When deploying models this is one of the most important things to think about.

Prompt injection is a concern anytime an external entity is given the ability to contribute to the prompt. 

