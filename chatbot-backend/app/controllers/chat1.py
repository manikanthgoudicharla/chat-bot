# services/chat_service.py
from sentence_transformers import SentenceTransformer
# import torch
from transformers import pipeline
# from transformers import AutoTokenizer, AutoModelForCausalLM
from app.config.db import index
# Load models once
embed_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

# tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
# model = AutoModelForCausalLM.from_pretrained(
#     "tiiuae/falcon-7b-instruct",
#     torch_dtype=torch.float16,
#     device_map="auto"
# )

text_generator = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", device_map="auto")

def generate_answer(prompt: str):
    vector = embed_model.encode(prompt).tolist()
    results = index.query(vector=vector, top_k=3, include_metadata=True)

    context = ""
    for match in results.matches:
        q = match.metadata.get("question", "")
        a = match.metadata.get("answer", "")
        context += f"Q: {q}\nA: {a}\n\n"

    final_prompt = f"{context}\nUser: {prompt}\nBot:"

    output = text_generator(final_prompt, max_new_tokens=100, temperature=0.7)[0]["generated_text"]
    return output.split("Bot:")[-1].strip()