# TinyGPT-2 — GPT-Style Language Model from Scratch

A minimal GPT-style autoregressive language model implemented **from scratch using PyTorch**. This project demonstrates the core components behind modern decoder-only Transformers, including **causal self-attention, multi-head attention, positional embeddings, feed-forward networks, Transformer blocks, SentencePiece tokenization, next-token prediction, and autoregressive text generation**.

The goal of this project is to understand the internal architecture and mathematics of GPT-style models by implementing the major components without relying on pretrained Transformer models.

---

## 🚀 Project Overview

**TinyGPT-2** is a small decoder-only Transformer language model trained on a custom text corpus.

The model learns to predict the **next token given previous tokens**, following the same fundamental autoregressive objective used by GPT-style language models.

For example:

```text
Input:
hello friends how are

Target:
friends how are you
```

During generation, the model predicts one token at a time:

```text
hello
      ↓
hello friends
      ↓
hello friends how
      ↓
hello friends how are
      ↓
hello friends how are you
```

This project focuses on understanding the architecture rather than achieving large-scale language modeling performance.

---

# 🧠 Architecture

The model follows a simplified GPT-style decoder architecture:

```text
                  Input Text
                      │
                      ▼
              SentencePiece
                 Tokenizer
                      │
                      ▼
                Token IDs
                      │
                      ▼
             Token Embeddings
                      │
                      +
                      │
             Positional Embeddings
                      │
                      ▼
        ┌─────────────────────────────┐
        │      Transformer Block      │
        │                             │
        │  LayerNorm                  │
        │      ↓                      │
        │  Causal Self-Attention      │
        │      ↓                      │
        │  Residual Connection        │
        │      ↓                      │
        │  LayerNorm                  │
        │      ↓                      │
        │  Feed Forward Network       │
        │      ↓                      │
        │  Residual Connection        │
        └─────────────────────────────┘
                      │
                    × N
                      │
                      ▼
                 LayerNorm
                      │
                      ▼
              Linear Projection
                      │
                      ▼
                    Logits
                      │
                      ▼
                 Softmax
                      │
                      ▼
              Next Token Prediction
```

---

# ✨ Key Features

- Transformer decoder architecture implemented from scratch
- Causal self-attention
- Multi-head self-attention
- Feed-forward neural network
- Layer normalization
- Residual connections
- Token embeddings
- Positional embeddings
- SentencePiece BPE tokenizer
- Autoregressive next-token prediction
- Cross-entropy language-modeling loss
- AdamW optimizer
- Temperature-independent multinomial sampling
- Custom text corpus
- Text generation from a user-provided prompt

---

# 🏗️ Model Configuration

The model is intentionally kept small so that it can be trained locally.

| Parameter | Value |
|---|---:|
| Vocabulary Size | 40 |
| Block Size | 6 |
| Embedding Dimension | 32 |
| Attention Heads | 2 |
| Transformer Layers | 2 |
| Learning Rate | 1e-3 |
| Training Steps | 1500 |
| Optimizer | AdamW |
| Activation | ReLU |
| Loss Function | Cross Entropy |
| Tokenizer | SentencePiece BPE |

---

# 🔤 Tokenization

The project uses **SentencePiece** with Byte Pair Encoding (BPE).

Instead of representing text directly as words, SentencePiece converts the corpus into subword tokens.

For example:

```text
"hello friends how are you"
```

may be represented internally as token IDs:

```text
[12, 7, 25, 18, 9, ...]
```

These integer IDs are then passed to PyTorch's embedding layer.

### Why SentencePiece?

Using a subword tokenizer provides several advantages:

- Handles unknown words better than word-level tokenization
- Produces a fixed vocabulary
- Can represent rare words using smaller subword units
- Works directly on raw text
- Similar tokenization principles are used in modern language models

The tokenizer is trained directly from:

```text
corpus.txt
```

using:

```python
spm.SentencePieceTrainer.Train(
    input="corpus.txt",
    model_prefix="tokenizer",
    vocab_size=40,
    model_type="bpe"
)
```

---

# 🔍 Self-Attention

The core of the Transformer is self-attention.

For an input representation \(X\), the model computes:

\[
Q = XW_Q
\]

\[
K = XW_K
\]

\[
V = XW_V
\]

The attention scores are calculated using:

\[
Attention(Q,K,V)
=
softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
\]

where:

- \(Q\) = Query matrix
- \(K\) = Key matrix
- \(V\) = Value matrix
- \(d_k\) = dimension of the key vectors

The implementation uses:

```python
wei = q @ k.transpose(-2, -1) / (C ** 0.5)
```

---

# 🔒 Causal Masking

Because this is an autoregressive language model, the model must not look at future tokens.

For example:

```text
The cat is
```

When predicting:

```text
cat
```

the model can only use:

```text
The
```

and cannot access future tokens.

A lower-triangular causal mask is therefore applied:

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

The implementation creates the mask using:

```python
self.register_buffer(
    'tril',
    torch.tril(torch.ones(block_size, block_size))
)
```

and applies it using:

```python
wei = wei.masked_fill(
    self.tril[:T, :T] == 0,
    float('-inf')
)
```

After softmax, the masked positions receive probability approximately equal to zero.

---

# 🧩 Multi-Head Attention

Instead of using a single attention mechanism, the model uses multiple attention heads.

With:

```text
embedding_dim = 32
n_heads = 2
```

each head operates on:

```text
32 / 2 = 16
```

dimensions.

The outputs of the individual heads are concatenated:

```python
out = torch.cat([h(x) for h in self.heads], dim=-1)
```

and projected back to the embedding dimension:

```python
return self.proj(out)
```

This allows different attention heads to learn different relationships between tokens.

---

# 🔄 Transformer Block

Each Transformer block contains:

1. Layer Normalization
2. Multi-Head Causal Self-Attention
3. Residual Connection
4. Layer Normalization
5. Feed-Forward Network
6. Residual Connection

The implementation follows the **Pre-LayerNorm** structure:

```python
x = x + self.sa(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```

Mathematically:

\[
x' = x + Attention(LN(x))
\]

followed by:

\[
x'' = x' + FFN(LN(x'))
\]

---

# ⚡ Feed-Forward Network

Each Transformer block contains a position-wise feed-forward neural network:

```python
self.net = nn.Sequential(
    nn.Linear(n_embd, 4 * n_embd),
    nn.ReLU(),
    nn.Linear(4 * n_embd, n_embd)
)
```

The dimensionality first expands:

\[
32 \rightarrow 128
\]

and then contracts:

\[
128 \rightarrow 32
\]

This gives the model additional nonlinear transformation capacity after the attention operation.

---

# 📍 Positional Embeddings

Self-attention itself does not inherently understand token order.

Therefore, the model adds learnable positional embeddings:

```python
self.position_embedding = nn.Embedding(
    block_size,
    embedding_dim
)
```

The token and position representations are added:

```python
x = tok_emb + pos_emb
```

Therefore:

\[
X = TokenEmbedding + PositionEmbedding
\]

This allows the model to distinguish between sequences such as:

```text
dog bites man
```

and:

```text
man bites dog
```

---

# 🎯 Training Objective

TinyGPT is trained using **next-token prediction**.

Given a sequence:

```text
hello friends how are you
```

the training pairs are shifted by one position.

### Input

```text
hello friends how are
```

### Target

```text
friends how are you
```

The model produces logits:

\[
Z \in \mathbb{R}^{B \times T \times V}
\]

where:

- \(B\) = batch size
- \(T\) = sequence length
- \(V\) = vocabulary size

Cross-entropy loss is then calculated:

\[
L =
-\frac{1}{N}
\sum_{i=1}^{N}
\log P(y_i|x_i)
\]

Implementation:

```python
loss = F.cross_entropy(
    logits.view(B*T, C),
    targets.view(B*T)
)
```

---

# 🏋️ Training

The model uses the AdamW optimizer:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr
)
```

Training consists of repeatedly:

```text
1. Sample a batch
       ↓
2. Create input/target sequences
       ↓
3. Forward pass
       ↓
4. Calculate cross-entropy loss
       ↓
5. Backpropagation
       ↓
6. Update parameters
```

Simplified training loop:

```python
for step in range(epochs):

    xb, yb = get_batch()

    logits, loss = model(xb, yb)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
```

Training progress is printed every 300 steps:

```text
Step 0, loss=...
Step 300, loss=...
Step 600, loss=...
Step 900, loss=...
Step 1200, loss=...
```

---

# ✍️ Text Generation

After training, the model can generate new text autoregressively.

The generation process is:

```text
Prompt
  ↓
Tokenize
  ↓
Transformer
  ↓
Predict next-token probabilities
  ↓
Sample next token
  ↓
Append token to sequence
  ↓
Repeat
```

The implementation:

```python
def generate(self, idx, max_new_tokens):

    for _ in range(max_new_tokens):

        idx_cond = idx[:, -block_size:]

        logits, _ = self(idx_cond)

        logits = logits[:, -1, :]

        probs = F.softmax(logits, dim=-1)

        next_idx = torch.multinomial(probs, 1)

        idx = torch.cat((idx, next_idx), dim=1)

    return idx
```

For example:

```python
context = torch.tensor(
    [sp.encode("hello")],
    dtype=torch.long
)

out = model.generate(
    context,
    max_new_tokens=20
)
```

The generated token IDs are finally decoded back into text:

```python
generated_ids = out[0].tolist()

print(sp.decode(generated_ids))
```

---

# 📚 Training Corpus

The current demonstration corpus contains simple sentences related to everyday topics:

```text
Hello friends how are you
the tea is very hot
my name is krishnandan
the roads of Delhi are busy
it is raining in Mumbai
the train is late again
i love eating samosas and drinking tea
holi is my favorite festival
diwali brings lights and sweets
india won the cricket match
```

Because the corpus is extremely small, this model is intended primarily for **educational and architectural understanding**, not general-purpose language generation.

---

# 📁 Project Structure

```text
tiny_gpt_2/
│
├── main.py
│
├── transformer_block.py
│
├── corpus.txt
│
├── README.md
│
└── .gitignore
```

### `main.py`

Contains:

- SentencePiece tokenizer training
- Dataset preparation
- Batch generation
- TinyGPT architecture
- Training loop
- Text generation

### `transformer_block.py`

Contains the core Transformer components:

- `SelfAttentionHead`
- `MultiHeadAttention`
- `FeedForward`
- `Block`

### `corpus.txt`

Contains the training text used to build the tokenizer and train the language model.

---

# 🛠️ Requirements

Recommended environment:

```text
Python 3.10+
PyTorch
SentencePiece
```

Install dependencies:

```bash
pip install torch sentencepiece
```

Or:

```bash
pip install -r requirements.txt
```

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/krishnadanp2057/tiny_gpt_2.git
```

Move into the project:

```bash
cd tiny_gpt_2
```

Install dependencies:

```bash
pip install torch sentencepiece
```

Run the model:

```bash
python main.py
```

---

# 💻 Hardware

This project is intentionally lightweight and can run on a CPU.

It can also take advantage of CUDA when available.

The code checks CUDA availability using:

```python
torch.cuda.is_available()
```

and reports the GPU using:

```python
torch.cuda.get_device_name(0)
```

For larger experiments, GPU acceleration is recommended.

---

# 📈 Limitations

This implementation is intentionally minimal and has several limitations compared with production-scale GPT models.

### Small model

The current configuration is:

```text
2 Transformer layers
2 attention heads
32-dimensional embeddings
40-token vocabulary
```

Therefore, the model has very limited representational capacity.

### Small dataset

The training corpus contains only a small number of sentences.

Consequently, the model cannot learn broad language understanding.

### No train/validation split

The current implementation trains directly on the complete corpus.

A production language model should use separate:

```text
Training set
Validation set
Test set
```

### No temperature parameter

Text generation currently uses:

```python
torch.multinomial(probs, 1)
```

without configurable temperature.

### No top-k/top-p sampling

More advanced generation strategies such as:

```text
Top-k sampling
Top-p / nucleus sampling
Temperature scaling
```

are not currently implemented.

### No learning-rate scheduler

The learning rate remains fixed during training.

### No dropout

Dropout is not currently used for regularization.

---

# 🔬 Concepts Demonstrated

This project provides hands-on implementation of several important concepts in modern NLP and LLM systems:

- Neural language modeling
- Tokenization
- BPE
- Embeddings
- Positional representations
- Query-Key-Value attention
- Scaled dot-product attention
- Causal masking
- Multi-head attention
- Residual connections
- Layer normalization
- Feed-forward networks
- Transformer architecture
- Autoregressive generation
- Cross-entropy loss
- Backpropagation
- AdamW optimization

---

# 🔮 Future Improvements

Possible extensions include:

- [ ] Add train/validation split
- [ ] Add configurable temperature
- [ ] Implement Top-k sampling
- [ ] Implement Top-p sampling
- [ ] Add dropout
- [ ] Add learning-rate scheduling
- [ ] Add model checkpointing
- [ ] Add validation loss
- [ ] Add perplexity evaluation
- [ ] Add GPU/device management
- [ ] Increase context length
- [ ] Increase model depth
- [ ] Increase embedding dimension
- [ ] Add attention visualization
- [ ] Add training-loss visualization
- [ ] Build an interactive text-generation interface
- [ ] Train on a larger corpus
- [ ] Implement mixed-precision training
- [ ] Implement weight tying
- [ ] Compare generated text with different sampling strategies

---

# 📖 Learning Objective

The primary objective of this project is **not to build a competitive language model**.

Instead, it is to understand how a GPT-style model works internally by implementing its major components from the ground up.

The project follows the fundamental flow:

```text
Raw Text
   ↓
SentencePiece Tokenization
   ↓
Token IDs
   ↓
Token Embeddings
   +
Positional Embeddings
   ↓
Transformer Blocks
   ↓
Causal Self-Attention
   ↓
Feed Forward Network
   ↓
Layer Normalization
   ↓
Linear Language Modeling Head
   ↓
Logits
   ↓
Next Token Prediction
   ↓
Autoregressive Generation
```

---

# 📚 References

This implementation is inspired by the core ideas introduced in the original Transformer and GPT architectures.

### Transformer

**Attention Is All You Need**

Vaswani et al., 2017.

### GPT

**Improving Language Understanding by Generative Pre-Training**

Radford et al., 2018.

### SentencePiece

SentencePiece — A simple and language-independent subword tokenizer and detokenizer for Neural Text Processing.

---

# 👨‍💻 Author

**Krishnandan Pandit**

M.Tech — Data Science

Interests:

- Machine Learning
- Deep Learning
- Natural Language Processing
- Large Language Models
- Generative AI
- Transformers

---

# ⭐ Acknowledgement

This project was developed as a hands-on implementation to understand the internal architecture of GPT-style Transformer language models.

The emphasis is on **learning by implementation** — building the fundamental components rather than treating the Transformer as a black box.

If you find this project useful for learning about Transformers and language models, consider giving the repository a ⭐.