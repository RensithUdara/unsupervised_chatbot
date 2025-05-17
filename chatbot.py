from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from database import get_conversations, log_conversation

# Initialize embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize conversational model
model_path = "./finetuned_model"
model_name = "facebook/blenderbot-400M-distill"

try:
    if os.path.exists(model_path):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("Loaded fine-tuned model.")
    else:
        raise FileNotFoundError("Fine-tuned model not found.")
except Exception as e:
    print(f"Failed to load fine-tuned model: {e}. Using default model.")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_response(user_input):
    # Tokenize input and generate response
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=4,
        length_penalty=0.6,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def get_cluster_response(user_input):
    # Load conversation data
    df = get_conversations()
    if df.empty or "cluster" not in df.columns or df["cluster"].isna().all():
        return generate_response(user_input)

    # Embed user input
    user_embedding = embedder.encode([user_input])

    # Embed stored inputs
    stored_inputs = df["user_input"].tolist()
    stored_embeddings = embedder.encode(stored_inputs)

    # Find closest cluster
    similarities = cosine_similarity(user_embedding, stored_embeddings)[0]
    closest_idx = np.argmax(similarities)
    closest_cluster = df.iloc[closest_idx]["cluster"]

    # Get a response from the same cluster
    cluster_responses = df[df["cluster"] == closest_cluster]["response"].tolist()
    if cluster_responses:
        return cluster_responses[0]
    return generate_response(user_input)

def get_response(user_input):
    response = get_cluster_response(user_input)
    log_conversation(user_input, response)
    return response

if __name__ == "__main__":
    from database import init_db
    init_db()  # Ensure database is initialized
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = get_response(user_input)
        print(f"Bot: {response}")