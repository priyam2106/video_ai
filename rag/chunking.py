def chunk_text_with_timestamps(
    segments,
    chunk_size=300,
    overlap=50
):
    """
    Create text chunks while preserving
    the start and end timestamps.
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

            # Keep overlap
            overlap_words = current_words[
                -overlap:
            ]

            current_words = overlap_words.copy()

            # The next chunk starts approximately
            # from the current segment
            current_start = segment["start"]

    # Store remaining words
    if current_words:

        chunks.append({
            "text": " ".join(current_words),
            "start_time": current_start,
            "end_time": current_end
        })

    return chunks