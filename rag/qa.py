from model_router import generate_with_router


def answer_question(question, retrieved_chunks,model_choice="auto"):

    if not retrieved_chunks:
        return "I could not find the answer in the provided videos."

    print("\n===== RETRIEVED CONTEXT =====")

    context_parts = []

    for i, result in enumerate(
        retrieved_chunks,
        start=1
    ):

        score = result.get("score", 0)

        start_time = result.get(
            "start_time",
            0
        )

        end_time = result.get(
            "end_time",
            0
        )

        video_name = result.get(
            "video_name",
            "Unknown video"
        )

        video_id = result.get(
            "video_id",
            "Unknown"
        )

        print(
            f"{i}. Video: {video_name} | "
            f"Score: {score:.4f} | "
            f"Start: {start_time:.2f}s | "
            f"End: {end_time:.2f}s"
        )

        context_parts.append(
            f"""
VIDEO: {video_name}
VIDEO ID: {video_id}
TIMESTAMP: {start_time:.2f} - {end_time:.2f} seconds

CONTENT:
{result["chunk"]}
"""
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are an AI teaching assistant that answers questions ONLY from the
provided video transcript context.

USER QUESTION:
{question}

RETRIEVED VIDEO CONTEXT:
--------------------------------
{context}
--------------------------------

RULES:

1. Answer ONLY using information present in the retrieved context.
2. Carefully read ALL retrieved chunks.
3. The transcript may contain speech-to-text errors.
4. Understand the intended meaning from surrounding sentences.
5. If the context directly or indirectly answers the question, give the answer.
6. Do not use outside knowledge.
7. Do not invent facts.
8. Keep the answer short and clear.
9. If multiple retrieved chunks contain relevant information, combine them.
10. Use the video information and timestamps only as supporting context.
11. If the context genuinely does not contain the answer, respond exactly:

I could not find the answer in the provided videos.

Now answer the question.
"""

    return generate_with_router(
    question,
    retrieved_chunks,
    prompt,
    model_choice
)