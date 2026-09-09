from rag.vector_store import (
    search_vector_store,
    search_all_videos
)


def semantic_search(query, video_id, top_k=5):

    return search_vector_store(
        query,
        video_id,
        top_k=top_k
    )


def global_semantic_search(query, video_ids, top_k=5):

    return search_all_videos(
        query,
        video_ids,
        top_k=top_k
    )