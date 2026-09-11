import ollama


# ==========================================
# AVAILABLE MODELS
# ==========================================

SMALL_MODEL = "llama3.2:3b"
BALANCED_MODEL = "qwen2.5:7b"
POWERFUL_MODEL = "llama3.1:8b"


AVAILABLE_MODELS = {
    "fast": SMALL_MODEL,
    "balanced": BALANCED_MODEL,
    "powerful": POWERFUL_MODEL
}


# ==========================================
# AUTOMATIC MODEL ROUTING
# ==========================================

def calculate_complexity(question, retrieved_chunks):
    score = 0

    question_lower = question.lower()

    if len(question) > 100:
        score += 15

    if len(question) > 200:
        score += 10

    if question.count("?") > 1:
        score += 15

    complex_words = [
        "explain",
        "compare",
        "difference",
        "analyze",
        "analysis",
        "why",
        "how does",
        "how do",
        "relationship",
        "advantages",
        "disadvantages",
        "summarize",
        "evaluate",
        "discuss",
        "reason",
        "infer",
        "conclusion"
    ]

    for word in complex_words:
        if word in question_lower:
            score += 8

    if len(retrieved_chunks) >= 3:
        score += 10

    if len(retrieved_chunks) >= 5:
        score += 10

    return min(score, 100)


def select_auto_model(question, retrieved_chunks):

    complexity = calculate_complexity(
        question,
        retrieved_chunks
    )

    if complexity >= 50:
        return POWERFUL_MODEL, complexity

    return SMALL_MODEL, complexity


# ==========================================
# MANUAL MODEL SELECTION
# ==========================================

def get_selected_model(model_choice, question, retrieved_chunks):

    # AUTO
    if model_choice == "auto":
        model, complexity = select_auto_model(
            question,
            retrieved_chunks
        )

        return model, complexity

    # USER SELECTED MODEL
    model = AVAILABLE_MODELS.get(
        model_choice
    )

    # Safety fallback
    if model is None:
        model = SMALL_MODEL
        model_choice = "fast"

    return model, None


# ==========================================
# GENERATE ANSWER
# ==========================================

def generate_with_router(
    question,
    retrieved_chunks,
    prompt,
    model_choice="auto"
):

    model, complexity = get_selected_model(
        model_choice,
        question,
        retrieved_chunks
    )

    print("\n===== AI MODEL SELECTION =====")

    if model_choice == "auto":

        print(
            f"Mode: AUTO | "
            f"Complexity: {complexity}/100"
        )

    else:

        print(
            f"Mode: MANUAL | "
            f"Selection: {model_choice}"
        )

    print(f"Selected Model: {model}")

    try:

        response = ollama.chat(
            model=model,
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

        return response["message"]["content"].strip()

    except Exception as e:

        print(
            f"Model '{model}' failed: {e}"
        )

        raise