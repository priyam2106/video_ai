import ollama

# ============================================

# Ollama Configuration

# ============================================

MODEL_NAME = "llama3.2:3b"

# ============================================

# Generate AI Summary

# ============================================

def generate_summary(transcript):
    """
    Generate a structured AI summary from
    a video transcript using Ollama.
    """

    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty.")

    prompt = f"""


You are an AI video summarization assistant.

Analyze the following video transcript carefully.

## TRANSCRIPT:

## {transcript}

Return the response using EXACTLY this format:

SUMMARY:
Write a concise 1-2 paragraph summary of the video.

KEY POINTS:

* Point 1
* Point 2
* Point 3
* Point 4
* Point 5

KEYWORDS:
keyword1, keyword2, keyword3, keyword4, keyword5

IMPORTANT RULES:

1. Keep the summary concise and informative.
2. Provide exactly 5 important key points.
3. Provide 5 to 10 important keywords.
4. Base the answer only on the provided transcript.
5. Do not invent information.
6. Do not add any extra sections.
7. Do not add headings other than:
   SUMMARY:
   KEY POINTS:
   KEYWORDS:
8. Keep the key points short and clear.
9. Separate keywords using commas.
    """

    print("\n====================================")
    print("Generating AI Summary using Ollama...")
    print("====================================")

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0
        }
    )

    answer = response["message"]["content"].strip()

    print("AI Summary generated successfully.")

    return answer

# ============================================

# Parse AI Summary

# ============================================

def parse_summary(summary_text):
    """
    Convert Ollama's structured response into
    separate summary, key points and keywords.
    """

    if not summary_text:
        return {
            "summary": "",
            "key_points": [],
            "keywords": []
        }

    summary = ""
    key_points = []
    keywords = []

    current_section = None

    for line in summary_text.splitlines():

        line = line.strip()

        # Ignore empty lines
        if not line:
            continue

        upper_line = line.upper()

        # ------------------------------------
        # SUMMARY
        # ------------------------------------

        if upper_line == "SUMMARY:":
            current_section = "summary"
            continue

    # ------------------------------------
    # KEY POINTS
    # ------------------------------------

        if upper_line == "KEY POINTS:":
            current_section = "key_points"
            continue

    # ------------------------------------
    # KEYWORDS
    # ------------------------------------

        if upper_line == "KEYWORDS:":
            current_section = "keywords"
            continue

    # ------------------------------------
    # Read Summary
    # ------------------------------------

        if current_section == "summary":

            if summary:
                summary += " "

            summary += line

    # ------------------------------------
    # Read Key Points
    # ------------------------------------

        elif current_section == "key_points":

            point = line.lstrip("-•* ").strip()

            if point:
                key_points.append(point)

    # ------------------------------------
    # Read Keywords
    # ------------------------------------

        elif current_section == "keywords":

            # Handle comma-separated keywords
            keyword_items = line.split(",")

            for keyword in keyword_items:

                keyword = keyword.strip()

                if keyword:
                    keywords.append(keyword)


    # Remove duplicate keywords
    unique_keywords = []

    for keyword in keywords:

        if keyword.lower() not in [
            item.lower()
            for item in unique_keywords
        ]:
            unique_keywords.append(keyword)


    return {
        "summary": summary.strip(),
        "key_points": key_points,
        "keywords": unique_keywords
    }


# ============================================

# Test Function

# ============================================

if __name__ == "__main__":


    print("Ollama service loaded successfully.")

    print("Model:", MODEL_NAME)

