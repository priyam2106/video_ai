from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

print("Loading embedding model...")

embedding_model = SentenceTransformer(MODEL_NAME)
print("Embedding model loaded successfully")

def create_embeddings(texts):
    """
    Convert a list of text chunks into embeddings.
    """

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True


    )
    return embeddings