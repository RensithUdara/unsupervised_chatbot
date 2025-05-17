from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import pandas as pd
from database import get_conversations, update_conversation_clusters

# Load the embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def update_clusters():
    # Load conversation data
    df = get_conversations()
    if df.empty:
        print("No conversation data yet.")
        return

    # Generate embeddings for user inputs
    inputs = df["user_input"].tolist()
    embeddings = embedder.encode(inputs)

    # Cluster inputs (e.g., 5 clusters)
    kmeans = KMeans(n_clusters=5, random_state=42)
    df["cluster"] = kmeans.fit_predict(embeddings)

    # Update database with cluster labels
    update_conversation_clusters(df)
    print("Clusters updated!")

    # Print cluster summaries (optional)
    for cluster in range(5):
        cluster_inputs = df[df["cluster"] == cluster]["user_input"].tolist()
        if cluster_inputs:
            print(f"Cluster {cluster}: {cluster_inputs[:3]}")

if __name__ == "__main__":
    update_clusters()