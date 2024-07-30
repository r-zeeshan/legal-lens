import streamlit as st
import json
import requests
from case_retrieval import retrieve_similar_cases
from sentence_transformers import SentenceTransformer
import gcsfs

### Getting the API KEYS from the environment
api_key = st.secrets["general"]["PINECONE_API_KEY"]
gcs_bucket = st.secrets["general"]["GCS_BUCKET"]
summarization_api_url = st.secrets["general"]["API_URL"]

### A default starting string
default_text = """
The plaintiff, John Doe, alleges that the defendant, Jane Smith, breached a contract by failing to deliver goods as agreed upon in a sales contract dated January 15, 2020. The plaintiff seeks damages for the loss incurred due to the non-delivery of goods. The key issues involve the interpretation of contract terms and the determination of whether the defendant's failure to deliver constitutes a breach of contract under the applicable law.
"""

# Initialize the Sentence-BERT model cached
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
    """
    Retrieves case summaries for a given input text.

    Args:
        input_text (str): The input text for which case summaries are to be retrieved.
        top_k (int, optional): The number of top case summaries to retrieve. Defaults to 5.

    Yields:
        dict: A dictionary containing the case ID and its corresponding summary.

    Raises:
        Exception: If there is an error processing a case ID.

    Returns:
        None
    """
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

input_text = st.text_area('Enter case details:', value=default_text, height=200)

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

if st.button('Clear'):
    input_text = ""

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
