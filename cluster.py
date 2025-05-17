from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

# Load the embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def update_clusters():
    # Load conversation data
    if not os.path.exists("conversations.csv"):
        print("No conversation data yet.")
        return

    df = pd.read_csv("conversations.csv")
    if df.empty:
        print("Conversation data is empty.")
        return

    # Generate embeddings for user inputs
    inputs = df["user_input"].tolist()
    embeddings = embedder.encode(inputs)

    # Cluster inputs (e.g., 5 clusters)
    kmeans = KMeans(n_clusters=5, random_state=42)
    df["cluster"] = kmeans.fit_predict(embeddings)

    # Save updated data with cluster labels
    df.to_csv("conversations.csv", index=False)
    print("Clusters updated!")

    # Print cluster summaries (optional)
    for cluster in range(5):
        cluster_inputs = df[df["cluster"] == cluster]["user_input"].tolist()
        if cluster_inputs:
            print(f"Cluster {cluster}: {cluster_inputs[:3]}")

if __name__ == "__main__":
    update_clusters()