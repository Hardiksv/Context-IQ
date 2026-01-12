import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
# Sample chunks (normally ye chunking step se aayenge)

chunks = [
    "Violence detection using deep learning models like CNN and LSTM.",
    "CCTV surveillance systems help improve public safety.",
    "Food delivery applications focus on logistics and user experience."
]

# Generate embeddings
embeddings = model.encode(chunks)

# Convert to numpy float32 (FAISS requirement)
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
dimension = embeddings.shape[1]  # 384
index = faiss.IndexFlatL2(dimension)

# Add vectors to index
index.add(embeddings)

print("Total vectors in FAISS index:", index.ntotal)

# -------- SEARCH TEST --------
query = "How is violence detected in CCTV footage?"
query_vector = model.encode([query]).astype("float32")

# Search top 2 nearest chunks
k = 2
distances, indices = index.search(query_vector, k)

print("\nQuery:", query)
print("\nTop results:")
for i in indices[0]:
    print("-", chunks[i])
