# Legal Lens - Precedence Lookup

## Project Overview

This project is a small legal aid natural language processing (NLP) tool designed to assist legal professionals in finding relevant case precedents and summaries. The tool is built using a subset of the Caselaw Access Project dataset, specifically the Caselaw Dataset (Illinois), which is available on Kaggle. 

The tool allows users to enter details about their current case and retrieves relevant cases from the dataset. It then provides a brief summary of each relevant case, helping legal professionals find past precedents and additional details to support their work.

## Dataset

### Caselaw Dataset (Illinois)

- **Source**: [Kaggle - Caselaw Dataset (Illinois)](https://www.kaggle.com/datasets/harvardlil/caselaw-dataset-illinois)
- **Context**: The Caselaw Access Project makes 40 million pages of U.S. caselaw freely available online from the collections of Harvard Law School Library.
  - Learn more: [Caselaw API](https://case.law/api/)
  - Access Limits: [API Limits](https://case.law/api/#limits)
- **Content**: This dataset includes all published U.S. caselaw from the state of Illinois (I.L.) in Text and XML format.
- **Acknowledgements**: The Caselaw Access Project is by the Library Innovation Lab at Harvard Law School Library.

## Features

- **Case Information Input**: Users can enter details about the case they are working on.
- **Relevant Case Retrieval**: The tool finds relevant cases from the dataset based on the entered details.
- **Case Summarization**: Provides a brief summary of each relevant case.
- **Precedent Support**: Helps legal professionals find past precedents and additional details to support their cases.

## Technologies Used

- **Python**: The core programming language used for development.
- **Streamlit**: Used to create a web application interface for the tool.
- **Transformers and Pretrained Models**: Utilized for NLP tasks such as text processing and summarization.
- **Google Cloud Storage**: Handles large datasets efficiently by storing and accessing data from the cloud.
- **Pinecone**: Used for similarity search and vector database.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/legal-lens.git
cd legal-lens
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the required dependencies:
```bash
pip install -r requirements.txts
```
4. Set up environment variables:

* Create a .env file in the root directory.
* Add your environment variables (API keys, bucket names, etc.) to the .env file.
Example .env file:

```bash
PINECONE_API_KEY=your_pinecone_api_key
GCS_BUCKET=your_gcs_bucket_name
```

## Usage
1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open the provided local URL in your web browser.

3. Enter the case details in the text area and select the number of similar cases to retrieve.

4. Click the "Find Similar Cases" button to retrieve and display relevant case summaries.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License.

## Acknowledgements
* The Caselaw Access Project by the Library Innovation Lab at Harvard Law School Library for providing the dataset.
* Streamlit for the easy-to-use web application framework.
* Hugging Face for the transformers and pretrained models.
* Google Cloud Storage for handling large datasets.
* Pinecone for similarity search and vector database capabilities.