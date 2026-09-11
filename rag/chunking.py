def chunk_text_with_timestamps(
    segments,
    chunk_size=300,
    overlap=50
):
    """
    Create text chunks while preserving
    accurate start and end timestamps.
    """

    chunks = []

    current_words = []
    current_start = None
    current_end = None

    for segment in segments:

        text = segment["text"].strip()

        if not text:
            continue

        words = text.split()

        # Set starting timestamp
        if current_start is None:
            current_start = segment["start"]

        # Add words
        current_words.extend(words)

        # Update ending timestamp
        current_end = segment["end"]

        # Create chunk when enough words are collected
        if len(current_words) >= chunk_size:

            chunks.append({
                "text": " ".join(current_words),
                "start_time": current_start,
                "end_time": current_end
            })

            # Keep overlap words
            overlap_words = current_words[-overlap:]

            current_words = overlap_words.copy()

            # We don't know the exact timestamp of the
            # first overlap word, so keep the previous
            # chunk start timestamp temporarily.
            current_start = None

    # Store remaining words
    if current_words:

        if current_start is None:
            # Fallback: use the last available segment start
            current_start = segments[-1]["start"]

        chunks.append({
            "text": " ".join(current_words),
            "start_time": current_start,
            "end_time": current_end
        })

    return chunks