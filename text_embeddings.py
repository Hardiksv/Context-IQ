from sentence_transformers import SentenceTransformer

# Load pre-trained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings(chunks):
    """
    Input: list of text chunks
    Output: list of vector embeddings
    """
    embeddings = model.encode(chunks)
    return embeddings


# # ---------- TEST ----------
# if __name__ == "__main__":
#     sample_chunks = [
#         "Violence detection using deep learning is an important research area.",
#         "CCTV surveillance systems help improve public safety."
#     ]

#     vectors = generate_embeddings(sample_chunks)

#     print("Total embeddings:", len(vectors))
#     print("Vector dimension:", len(vectors[0]))
