from sentence_transformers import SentenceTransformer
from transformers import pipeline
from app.config.db import index
import torch

# Load embedding model
embed_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load GPT-2 model for generation
text_generator = pipeline(
    "text-generation",
    model="gpt2",
    torch_dtype=torch.float32,
    device=0 if torch.cuda.is_available() else -1
)

def generate_answer(prompt: str) -> str:
    # Embed user input
    vector = embed_model.encode(prompt).tolist()

    # Query top relevant content
    results = index.query(
        vector=vector,
        top_k=3,  # Use more matches to improve response diversity
        include_metadata=True,
        include_values=False
    )

    # Prepare context
    context_blocks = []
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
            context_blocks.append(f"User asked: {question}\nBot answered: {answer}")
        elif algorithm and explanation:
            context_blocks.append(f"Topic: {algorithm}\nDetails: {explanation}")

    context = "\n\n".join(context_blocks)

    # Final GPT prompt with refined format
    final_prompt = f"""You're a friendly and intelligent assistant 🤖.
Answer the user's message using the helpful context below.

Context:
{context if context else 'No prior context available.'}

Now respond naturally and clearly.

User: {prompt}
Bot:"""

    # Generate response
    output = text_generator(
        final_prompt,
        max_length=120,
        do_sample=True,
        temperature=0.7,
        pad_token_id=50256
    )[0]["generated_text"]

    # Extract just the bot's part
    if "Bot:" in output:
        response = output.split("Bot:")[-1].strip()
    else:
        response = output.strip()

    # Clean up long response (optional)
    if "." in response:
        response = response.split(".")[0].strip() + "."

    return response
