import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from transformer_block import Block, MultiHeadAttention, FeedForward


print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")


import sentencepiece as spm

with open("/Users/krishnandanpandit/Desktop/LLM/L2_tiny_GPT_2/corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

spm.SentencePieceTrainer.Train(
    input="/Users/krishnandanpandit/Desktop/LLM/L2_tiny_GPT_2/corpus.txt",
    # Prefix used for the generated tokenizer files
    # This will create files such as:
    # tokenizer.model
    # tokenizer.vocab
    model_prefix="tokenizer",
      # Number of tokens we want in our vocabulary
    vocab_size=40,   
    # Use BPE (Byte Pair Encoding) as the tokenization algorithm  
    model_type="bpe"
)

# Create a SentencePiece tokenizer/processor object
# This object will be used to encode and decode text
sp = spm.SentencePieceProcessor()
# Load the trained tokenizer model
sp.load("tokenizer.model")
# Convert the complete text into token IDs
# out_type=int means the output will be integer IDs
#
# Example:
# "hello world"
#       ↓
# [12, 7, 25, ...]
ids = sp.encode(text, out_type=int) 

# Convert the list of token IDs into a PyTorch tensor
# dtype=torch.long is required because token IDs are used
# as indices in nn.Embedding
data = torch.tensor(ids, dtype=torch.long) 

print(data)

vocab_size = sp.get_piece_size() 
# Get the actual vocabulary size from the trained tokenizer
# This tells us how many unique tokens the SentencePiece
# model contains
print(vocab_size)

# corpus = [
#     "hello friends how are you",
#     "the tea is very hot",
#     "my name is Aarohi",
#     "the roads of Delhi are busy",
#     "it is raining in Mumbai",
#     "the train is late again",
#     "i love eating samosas and drinking tea",
#     "holi is my favorite festival",
#     "diwali brings lights and sweets",
#     "india won the cricket match"
# ]

# corpus = [s + " <END>" for s in corpus]
# text = " ".join(corpus)
# print (text)

# words = list(set(text.split()))
# print(words)

# vocab_size = len(words)
# print(vocab_size)

# word2idx = {w: i for i, w in enumerate(words)}
# print("word2idx : ", word2idx)

# idx2word = {i: w for w, i in word2idx.items()} 
# print("idx2word : ", idx2word) 

# data = torch.tensor([word2idx[w] for w in text.split()], dtype=torch.long)
# print("data : ", data) 
# print(len(data))

block_size = 6 
embedding_dim = 32
n_heads = 2  
n_layers = 2 
lr = 1e-3 
epochs = 1500  


def get_batch(batch_size=16):
    ix = torch.randint(len(data) - block_size, (batch_size,))  
    x = torch.stack([data[i:i+block_size] for i in ix])  
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) 
    return x, y




class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim) 

        self.position_embedding = nn.Embedding(block_size, embedding_dim) 
        self.blocks = nn.Sequential(*[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)]) 

        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size) 

    def forward(self, idx, targets=None):
        B, T = idx.shape 
        tok_emb = self.token_embedding(idx) 
        
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb  
        x = self.blocks(x) 
        x = self.ln_f(x)
        logits = self.head(x) 
        loss = None
        if targets is not None:
            B, T, C = logits.shape 
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T)) 
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx



model = TinyGPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for step in range(epochs):
    xb, yb = get_batch() 
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 300 == 0:
        print(f"Step {step}, loss={loss.item():.4f}")



#context = torch.tensor([[word2idx["hello"]]], dtype=torch.long)
#out = model.generate(context, max_new_tokens=15)

#print("\nGenerated text:\n")
#print(" ".join(idx2word[int(i)] for i in out[0]))

import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")

context = torch.tensor([sp.encode("hello")], dtype=torch.long)

out = model.generate(context, max_new_tokens=20)

print("\nGenerated text:\n")
# print(" ".join(idx2word[int(i)] for i in out[0]))

generated_ids = out[0].tolist()
print(sp.decode(generated_ids))