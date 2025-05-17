from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import os

# Initialize models
chatbot = pipeline("text2text-generation", model="facebook/blenderbot-400M-distill")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
CONVERSATION_LOG = "conversations.csv"

def log_conversation(user_input, response):
    data = {"user_input": user_input, "response": response}
    df = pd.DataFrame([data])
    if os.path.exists(CONVERSATION_LOG):
        df.to_csv(CONVERSATION_LOG, mode='a', header=False, index=False)
    else:
        df.to_csv(CONVERSATION_LOG, index=False)

def get_cluster_response(user_input):
    if not os.path.exists(CONVERSATION_LOG):
        return chatbot(user_input)[-1]['generated_text']

    # Load conversation data
    df = pd.read_csv(CONVERSATION_LOG)
    if df.empty or "cluster" not in df.columns:
        return chatbot(user_input)[-1]['generated_text']

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
        return cluster_responses[0]  # Return first response in cluster
    return chatbot(user_input)[-1]['generated_text']

def get_response(user_input):
    response = get_cluster_response(user_input)
    log_conversation(user_input, response)
    return response

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = get_response(user_input)
        print(f"Bot: {response}")
