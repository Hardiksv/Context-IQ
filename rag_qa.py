import os
import faiss
import numpy as np
from dotenv import load_dotenv
from github import Github
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


# ============================================================
# STEP 3.2 — REPO SELECTION (USER INTERACTION LAYER)
# ============================================================
def select_github_repo():
    """
    Shows GitHub repositories and lets the user select one.
    Returns the selected repository name.
    """
    load_dotenv()
    token = os.getenv("GITHUB_TOKEN")

    if not token:
        raise ValueError("GITHUB_TOKEN not found in .env file")

    g = Github(token)
    user = g.get_user()
    repos = list(user.get_repos())

    if not repos:
        raise ValueError("No repositories found in GitHub account")

    print("\nAvailable GitHub Repositories:\n")

    for i, repo in enumerate(repos, start=1):
        print(f"{i}. {repo.name}")

    while True:
        try:
            choice = int(input("\nSelect repository number: "))
            if 1 <= choice <= len(repos):
                selected_repo = repos[choice - 1].name
                print(f"\nSelected Repository: {selected_repo}\n")
                return selected_repo
            else:
                print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a valid number.")


# ============================================================
# GENERATION LAYER (FREE, RULE-BASED — NO LLM)
# ============================================================
def generate_answer_from_context(question, chunks):
    """
    Converts retrieved chunks into a readable answer.
    This is a placeholder generation layer (LLM will replace later).
    """

    combined_text = " ".join(chunks).lower()
    question_lower = question.lower()

    # Example: Resume Analyzer
    if "resume" in question_lower and "flask" in combined_text:
        return (
            "The Resume Analyzer is a Flask-based web application that allows users "
            "to upload resumes and compare them with job descriptions. "
            "It extracts skills from resumes, calculates a match score, "
            "and highlights missing skills to help improve job fit."
        )

    # Generic fallback
    answer = "Relevant information found:\n"
    for chunk in chunks[:2]:
        answer += f"- {chunk[:200]}...\n"

    return answer


# ============================================================
# RAG QUERY FUNCTION
# ============================================================
def ask_question(question, index, all_chunks):
    """
    Performs semantic search and generates an answer.
    """
    query_vector = embedding_model.encode([question]).astype("float32")
    _, indices = index.search(query_vector, TOP_K)

    retrieved_chunks = [all_chunks[i] for i in indices[0]]
    return generate_answer_from_context(question, retrieved_chunks)


# ============================================================
# MAIN EXECUTION (AUTOMATED FLOW)
# ============================================================
if __name__ == "__main__":

    # 1️⃣ Repo selection
    selected_repo = select_github_repo()

    # 2️⃣ Load data ONLY for selected repo
    github_texts = load_github_texts(selected_repo)
    drive_texts = load_drive_texts()  # optional; can disable later

    all_texts = github_texts + drive_texts
    print(f"Loaded documents: {len(all_texts)}")

    # 3️⃣ Chunking
    all_chunks = []
    for text in all_texts:
        all_chunks.extend(chunk_text(text, CHUNK_SIZE, OVERLAP))

    print(f"Total chunks created: {len(all_chunks)}")

    # 4️⃣ Embeddings
    embeddings = embedding_model.encode(all_chunks)
    embeddings = np.array(embeddings).astype("float32")

    # 5️⃣ FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(f"FAISS index ready with {index.ntotal} vectors")

    # 6️⃣ Question loop
    while True:
        question = input("\nAsk a question about this repository (or type 'exit'): ")
        if question.lower() == "exit":
            break

        answer = ask_question(question, index, all_chunks)
        print("\nAnswer:\n", answer)
