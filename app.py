import streamlit as st
import json
import requests
from case_retrieval import retrieve_similar_cases
from sentence_transformers import SentenceTransformer
import gcsfs

api_key = st.secrets["general"]["PINECONE_API_KEY"]
gcs_bucket = st.secrets["general"]["GCS_BUCKET"]
summarization_api_url = st.secrets["general"]["API_URL"]

# Initialize the Sentence-BERT model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Load specific case JSON from Google Cloud Storage
def load_case_json(case_id):
    gcs_file_path = f'gs://{gcs_bucket}/cases//case_{case_id}.json'
    fs = gcsfs.GCSFileSystem()
    with fs.open(gcs_file_path, 'r') as f:
        case_data = json.load(f)
    return case_data

# Summarize text using the backend API
def summarize_text(text):
    response = requests.post(summarization_api_url, json={"text": text})
    if response.status_code == 200:
        return response.json().get("summary")
    else:
        st.error("Error in summarizing text")
        return ""

def get_case_summaries(input_text, top_k=5):
    similar_case_ids = retrieve_similar_cases(input_text, top_k=top_k)
    summaries = []

    for case_id in similar_case_ids:
        try:
            case_id_int = int(float(case_id))
            case_data = load_case_json(case_id_int)
            case_text = case_data['majority_opinion']
            summary = summarize_text(case_text)
            summaries.append({'case_id': case_id_int, 'summary': summary})
            yield {'case_id': case_id_int, 'summary': summary}
        except Exception as e:
            st.error(f"Error processing case_id {case_id}: {e}")



st.title('Legal Document Analysis')
st.header('Find Similar Cases and Summarize Them')

input_text = st.text_area('Enter case details:', height=200)

top_k = st.slider('Number of similar cases to retrieve:', 1, 10, 5)

if st.button('Find Similar Cases'):
    if input_text.strip() == "":
        st.error("Please enter the case details.")
    else:
        st.info('Retrieving similar cases and generating summaries...')
        summary_container = st.container()
        
        for i, case in enumerate(get_case_summaries(input_text, top_k=top_k)):
            with summary_container:
                st.expander(f'Case {i+1} (ID: {case["case_id"]})').write(case['summary'])

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
