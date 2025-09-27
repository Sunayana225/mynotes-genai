# Personal Study Notes - Oracle Gen AI Course

**Disclaimer:** These are my personal study notes for the Oracle Gen AI course and are not official course material. They are for educational purposes and personal reference only.

---

## LLM - Large Language Model

Language model is very probabilistic model of text. Large in large language model (LLM) refers to # of parameters (no. of parameters); no agreed upon threshold.

## LLM Architectures

### Encoders and Decoders

Multiple architectures built on encoding and decoding focus on embedding and text generation. Embedding is a way of representing words, tokens or entire pieces of text as numerical vectors so that the neural network can understand and process them.

### What an Embedding Represents

- Each token (word, subword, or character) is turned into a vector of, say, 768 or 1024 numbers
- The position of the vector in this space captures semantic relationships
- Words with similar meaning end up closer together
- e.g., the embeddings for "king" and "queen" are closer than for "king" and "banana"

All models are built on transformer architecture. Each type of model has different capabilities (embedding or generation). It encoders and decoders have different sizes.

In the realm of language model size = no. of trainable parameters that model has.

![alt text](image.png)

Decoders are relatively large compare to encoders. When models are too small they have proven to be bad text generators. With a bit of added cleverness we may generate better text.

**Refer to the pdf to have more content on this topic**

## Encoders

![alt text](image-1.png)

Models that convert a sequence of words to an embedding vector representation. These vector representations are designed to be consumed by later models to do things like classification or regression.

But a lot of their use these days is for semantic search or vector search in databases. Let's say we are wanting to retrieve a document from a corpus:

1. To accomplish this you could encode or synonymously embed each doc in a corpus and store them in an index
2. When you get an input snippet you encode that too and check the similarity with the encoded input against the similarity of each document of the corpus and return the most similar one

## Decoders

Models take a sequence of words and output next word.

**Examples:** GPT 4, Llama, bloom, cohere

What these models do is they take a sequence of tokens and emit the next token in the sequence based on the probability of the vocabulary which they compute.

- A decoder produces only single token at a time
- We can always invoke a decoder over and over to generate many new tokens as we want
- To do that we need to first feed in a sequence of tokens and invoke model to produce the next token then append the generated token to generate the input sequence and feed it back to the model so that it can produce the second token
- It is a very expensive process and typically we would not use decoder model for embedding
- Decoders are being used now for generation text as they have shown tremendous amounts of capability to do so

## Encoder-Decoder

Encodes a specific sequence of words and use the encoding + to output a next word.

![alt text](image-2.png)
![alt text](image-3.png)
