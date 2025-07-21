from sentence_transformers import SentenceTransformer
from transformers import pipeline
from app.config.db import index
import torch

embed_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

text_generator = pipeline(
    "text-generation",
    model="gpt2",
    torch_dtype=torch.float32,
    device=0 if torch.cuda.is_available() else -1
)

def generate_answer(prompt: str) -> str:
    vector = embed_model.encode(prompt).tolist()
    results = index.query(vector=vector, top_k=3, include_metadata=True)

    context = ""
    seen = set()
    
    for match in results.matches:
        question = match.metadata.get("question")
        answer = match.metadata.get("answer")
        explanation = match.metadata.get("text")
        algorithm = match.metadata.get("algorithm")

        unique_key = f"{question}:{answer}:{algorithm}:{explanation}"
        if unique_key in seen:
            continue
        seen.add(unique_key)

        if question and answer:
            context += f"Q: {question}\nA: {answer}\n\n"
        elif explanation and algorithm:
            context += f"Algorithm: {algorithm}\nExplanation: {explanation}\n\n"

    max_context_length = 1500
    if len(context) > max_context_length:
        context = context[:max_context_length]

    final_prompt = f"""You are a helpful AI tutor 🤖.
Use the following information to answer the user's question briefly and clearly.

{context}
User: {prompt}
Bot:"""

    output = text_generator(
        final_prompt,
        max_length=100,
        do_sample=True,
        temperature=0.7,
        pad_token_id=50256
    )[0]["generated_text"]

    response = output.split("Bot:")[-1].strip()

    return response
