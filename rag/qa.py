import ollama

MODEL_NAME = "llama3.2:3b"


def answer_question(question, context):
    prompt = f"""
You are an AI teaching assistant for educational videos.

You will receive:
1. A user's question
2. Text retrieved from one or more videos

Your job is to answer the question using the retrieved text.

IMPORTANT:
- Carefully read the entire retrieved context.
- If the context contains information that answers the question, answer it.
- Do NOT say that the answer is missing if the context provides relevant information.
- Do NOT use outside knowledge.
- Do NOT invent facts.
- Give a short and clear answer.
- The retrieved text may contain speech-to-text errors. Understand the intended meaning from the surrounding words.

USER QUESTION:
{question}

RETRIEVED VIDEO CONTEXT:
-------------------------
{context}
-------------------------

Now answer the user's question.

If the retrieved context genuinely does not contain the answer, respond exactly:
I could not find the answer in the provided videos.
"""

    print("\n===== OLLAMA PROMPT =====")
    print(prompt)
    print("=========================\n")

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

    return answer