import os

import chromadb

from config import BASE_DIR
from rag.embeddings import create_embeddings


# ==========================================
# ChromaDB Configuration
# ==========================================

CHROMA_FOLDER = os.path.join(
    BASE_DIR,
    "chroma_db"
)

client = chromadb.PersistentClient(
    path=CHROMA_FOLDER
)

collection = client.get_or_create_collection(
    name="video_transcripts",
    metadata={
        "hnsw:space": "cosine"
    }
)


# ==========================================
# Create Vector Store
# ==========================================

def create_vector_store(chunks, video_id):

    if not chunks:
        raise ValueError("No text chunks provided.")

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # Create embeddings
    embeddings = create_embeddings(texts)

    # Unique IDs for each chunk
    ids = [
        f"video_{video_id}_chunk_{i}"
        for i in range(len(chunks))
    ]

    # Store timestamp information
    metadatas = [
        {
            "video_id": str(video_id),
            "chunk_number": i,
            "start_time": float(chunk["start_time"]),
            "end_time": float(chunk["end_time"])
        }
        for i, chunk in enumerate(chunks)
    ]

    # Remove old vectors for this video
    try:
        collection.delete(
            where={
                "video_id": str(video_id)
            }
        )
    except Exception:
        pass

    # Add new vectors
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print(
        f"Vector store created for video {video_id}"
    )

    print(
        f"Stored {len(chunks)} chunks."
    )

    return True


# ==========================================
# Search Within One Video
# ==========================================

def search_vector_store(
    query,
    video_id,
    top_k=5
):

    # Create query embedding
    query_embedding = create_embeddings(
        [query]
    )

    # Search ChromaDB
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        where={
            "video_id": str(video_id)
        }
    )

    retrieved_chunks = []

    documents = results.get(
        "documents",
        [[]]
    )[0]

    distances = results.get(
        "distances",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    # Process results
    for document, distance, metadata in zip(
        documents,
        distances,
        metadatas
    ):

        similarity = 1 - distance

        retrieved_chunks.append(
            {
                "chunk": document,
                "score": float(similarity),
                "start_time": metadata.get(
                    "start_time"
                ),
                "end_time": metadata.get(
                    "end_time"
                ),
                "metadata": metadata
            }
        )

    return retrieved_chunks


# ==========================================
# Search Across Multiple Videos
# ==========================================

def search_all_videos(
    query,
    video_ids,
    top_k=5
):

    if not video_ids:
        return []

    # Create query embedding
    query_embedding = create_embeddings(
        [query]
    )

    # Search all selected videos
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        where={
            "video_id": {
                "$in": [
                    str(video_id)
                    for video_id in video_ids
                ]
            }
        }
    )

    retrieved_chunks = []

    documents = results.get(
        "documents",
        [[]]
    )[0]

    distances = results.get(
        "distances",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    for document, distance, metadata in zip(
        documents,
        distances,
        metadatas
    ):

        similarity = 1 - distance

        retrieved_chunks.append(
            {
                "chunk": document,
                "score": float(similarity),
                "video_id": int(
                    metadata["video_id"]
                ),
                "start_time": float(
                    metadata.get(
                        "start_time",
                        0
                    )
                ),
                "end_time": float(
                    metadata.get(
                        "end_time",
                        0
                    )
                )
            }
        )
    # ============================================
    # Remove duplicate / very similar chunks
    # ============================================

    unique_chunks = []
    seen_text = set()

    for result in retrieved_chunks:

        text = result["chunk"].strip().lower()

        text_key = text[:150]

        if text_key not in seen_text:
            seen_text.add(text_key)
            unique_chunks.append(result)

    retrieved_chunks = unique_chunks


# ============================================
# Sort by relevance
# ============================================

    retrieved_chunks.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return retrieved_chunks


# ==========================================
# Delete Video Vectors
# ==========================================

def delete_video_vectors(video_id):

    try:

        collection.delete(
            where={
                "video_id": str(video_id)
            }
        )

        print(
            f"Deleted ChromaDB data for video {video_id}"
        )

    except Exception as e:

        print(
            f"ChromaDB delete error: {e}"
        )