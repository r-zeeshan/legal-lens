import pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_csv("cleaned_data.csv")

embeddings = model.encode(df['majority_opinion'].to_list(), show_progress_bar=True)
embeddings_df = pd.DataFrame(embeddings)
embeddings_df['id'] = df['id']

embeddings_df.to_csv('caselaw_embeddings.csv', index=False)