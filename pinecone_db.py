import os
import streamlit as st
import pandas as pd
from pinecone import Pinecone, ServerlessSpec

api_key = st.secrets["general"]["PINECONE_API_KEY"]

pc = Pinecone(api_key=api_key)

# Connect to the index
index_name = 'caselaw-index'

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Dimension of Sentence-BERT embeddings
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    )

index = pc.Index(index_name)

index_stats = index.describe_index_stats()
print(index_stats)

vector_count = index_stats['total_vector_count']
print(f"Current vector count in the index: {vector_count}")


# Load embeddings
embeddings_file_path = 'caselaw_embeddings.csv'
embeddings_df = pd.read_csv(embeddings_file_path)


# Calculate the starting point for the next batch
batch_size = 100
start_index = vector_count

for i in range(start_index, len(embeddings_df), batch_size):
    batch = embeddings_df.iloc[i:i+batch_size]
    vectors = [{"id": str(row['id']), "values": row[:-1].tolist()} for _, row in batch.iterrows()]
    index.upsert(vectors=vectors)

print("Embeddings uploaded to Pinecone successfully")
