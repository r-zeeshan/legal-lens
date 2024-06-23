import streamlit as st
import pandas as pd
from case_retrieval import retrieve_similar_cases
from text_summarization import summarize_text
from sentence_transformers import SentenceTransformer
import gcsfs

# Load secrets
api_key = st.secrets["PINECONE_API_KEY"]
gcs_bucket = st.secrets["GCS_BUCKET"]

# Load dataset from Google Cloud Storage
@st.cache_data
def load_dataset_chunked():
    gcs_file_path = f'gs://{gcs_bucket}/cleaned_data.csv'
    fs = gcsfs.GCSFileSystem()
    return pd.read_csv(gcs_file_path, storage_options={'gcsfs': fs}, chunksize=10000)

# Initialize the Sentence-BERT model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def get_case_summaries(input_text, top_k=5):
    similar_case_ids = retrieve_similar_cases(input_text, top_k=top_k)
    summaries = []
    
    df_iterator = load_dataset_chunked()
    for df_chunk in df_iterator:
        case_lookup = {int(row['id']): row['majority_opinion'] for _, row in df_chunk.iterrows()}
        
        for case_id in similar_case_ids:
            try:
                case_id_int = int(float(case_id))
                if case_id_int in case_lookup:
                    case_text = case_lookup[case_id_int]
                    summary = summarize_text(case_text)
                    summaries.append({'case_id': case_id_int, 'summary': summary})
            except Exception as e:
                st.error(f"Error processing case_id {case_id}: {e}")
                
        if len(summaries) >= top_k:
            break
    
    return summaries[:top_k]

# Streamlit app layout
st.title('Legal Document Analysis')
st.header('Find Similar Cases and Summarize Them')

input_text = st.text_area('Enter case details:', height=200)

top_k = st.slider('Number of similar cases to retrieve:', 1, 10, 5)

# Button to trigger case retrieval and summarization
if st.button('Find Similar Cases'):
    if input_text.strip() == "":
        st.error("Please enter the case details.")
    else:
        with st.spinner('Retrieving similar cases and generating summaries...'):
            case_summaries = get_case_summaries(input_text, top_k=top_k)
        st.success('Similar cases and summaries retrieved successfully!')
        
        for i, case in enumerate(case_summaries):
            with st.expander(f'Case {i+1} (ID: {case["case_id"]})'):
                st.write(case['summary'])

# Footer
st.markdown(
    """
    <style>
    footer {visibility: hidden;}
    .footer {visibility: visible; position: relative; bottom: 10px; text-align: center;}
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<div class="footer">Developed by Zeeshan Hameed</div>', unsafe_allow_html=True)
