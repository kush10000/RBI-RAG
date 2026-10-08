from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")  # first run downloads ~130 MB

sentences = [
    "KYC norms for banks",
    "Know Your Customer requirements for regulated entities",
    "Monsoon rainfall in Kerala",
]
vecs = model.encode(sentences, normalize_embeddings=True)

print(vecs.shape)       # (3, 384): 3 sentences, 384 numbers each
print(vecs @ vecs.T)    # similarity of every pair; 1.0 = identical meaning