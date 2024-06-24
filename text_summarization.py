from transformers import T5ForConditionalGeneration, T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-large')
model = T5ForConditionalGeneration.from_pretrained('t5-large')

def summarize_text(text):
    """
    Summarize the given text using the T5 model.
    
    Parameters:
    text (str): The text to summarize.
    
    Returns:
    str: The generated summary.
    """
    inputs = tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=512, truncation=True)
    
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

tokenizer.save_pretrained('t5-large-tokenizer')
model.save_pretrained('t5-large-model')