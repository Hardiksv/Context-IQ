def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks


# sample_text = """
# Violence detection using deep learning is an important area of research...
# """ * 10

# chunks = chunk_text(sample_text)

# print("Total chunks:", len(chunks))
# print("\nFirst chunk:\n", chunks[0])
