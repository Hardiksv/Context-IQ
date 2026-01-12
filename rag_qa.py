import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -------- IMPORT LOADERS & CHUNKER --------
from drive_text_loader import load_drive_texts
from git_hub_integration import load_github_texts
from text_chunker import chunk_text

# -------- CONFIG --------
CHUNK_SIZE = 500
OVERLAP = 100
TOP_K = 3

# -------- EMBEDDING MODEL --------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------- 1) LOAD REAL DATA --------
drive_texts = load_drive_texts()
github_texts = load_github_texts()

all_texts = drive_texts + github_texts
print(f"Loaded texts -> Drive: {len(drive_texts)}, GitHub: {len(github_texts)}")

# -------- 2) CHUNKING --------
all_chunks = []
for text in all_texts:
    all_chunks.extend(chunk_text(text, CHUNK_SIZE, OVERLAP))

print(f"Total chunks created: {len(all_chunks)}")

# -------- 3) EMBEDDINGS --------
embeddings = embedding_model.encode(all_chunks)
embeddings = np.array(embeddings).astype("float32")

# -------- 4) FAISS INDEX --------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"FAISS index ready with {index.ntotal} vectors")

# -------- 5) GENERATION LOGIC (FREE, NO LLM) --------
def generate_answer_from_context(question, chunks):
    """
    Simple rule-based generation layer.
    Converts retrieved context into a human-readable answer.
    """

    combined_text = " ".join(chunks).lower()
    question_lower = question.lower()

    # Resume Analyzer project
    if "resume analyzer" in question_lower or "resume" in question_lower:
        if "flask" in combined_text:
            return (
                "The Resume Analyzer is a Flask-based web application that allows users "
                "to upload resumes and compare them with job descriptions. "
                "It extracts skills from resumes, calculates a match score, "
                "and highlights missing skills to help improve job fit."
            )

    # Generic fallback
    summary = "Here is the relevant information found:\n"
    for chunk in chunks[:2]:  # limit verbosity
        summary += f"- {chunk[:200]}...\n"

    return summary


# -------- 6) RAG QUERY FUNCTION --------
def ask_question(question, top_k=TOP_K):
    # Embed the query
    q_vec = embedding_model.encode([question]).astype("float32")

    # Retrieve relevant chunks
    _, idx = index.search(q_vec, top_k)
    retrieved_chunks = [all_chunks[i] for i in idx[0]]

    # Generate final answer
    return generate_answer_from_context(question, retrieved_chunks)


# -------- TEST --------
if __name__ == "__main__":
    query = "What does the Resume Analyzer project do?"
    print("\nQuestion:", query)
    print("\nAnswer:\n", ask_question(query))
