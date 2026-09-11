import ollama

MODEL_NAME = "bge-m3"


def create_embeddings(texts):

    if not texts:
        return []

    print("Creating embeddings using Ollama bge-m3...")

    response = ollama.embed(
        model=MODEL_NAME,
        input=texts
    )

    embeddings = response["embeddings"]

    print(f"Created {len(embeddings)} embeddings.")

    return embeddings


def create_query_embedding(query):

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    response = ollama.embed(
        model=MODEL_NAME,
        input=query
    )

    return response["embeddings"][0]