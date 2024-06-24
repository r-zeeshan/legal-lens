from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import streamlit as st

# import os
# from dotenv import load_dotenv
# load_dotenv()
# api_key = os.environ.get('PINECONE_API_KEY')

api_key = st.secrets["general"]["PINECONE_API_KEY"]

pc = Pinecone(api_key=api_key)
index_name = 'caselaw-index'
index = pc.Index(index_name)

model = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve_similar_cases(input_text, top_k=5):
    """
    Generate embeddings for the input text and retrieve similar cases from Pinecone.
    
    Parameters:
    input_text (str): The input text to find similar cases for.
    top_k (int): The number of similar cases to retrieve.
    
    Returns:
    list: A list of similar case IDs.
    """
    input_embedding = model.encode([input_text])[0]
    
    query_result = index.query(
        vector=input_embedding.tolist(),
        top_k=top_k,
        include_values=False
    )
    
    similar_case_ids = [match['id'].split('.')[0] for match in query_result['matches']]
    return similar_case_ids
