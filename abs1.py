from transformers import BartTokenizer, BartForConditionalGeneration
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize


model_name = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def chunk_text(text, chunk_size=512):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def summarize_text(text, num_sentences):
   chunks = chunk_text(text, chunk_size=512)
   summaries = []
    for chunk in chunks:
       
        inputs = tokenizer([chunk], max_length=512, return_tensors='pt', truncation=True)
        summary_ids = model.generate(inputs['input_ids'], max_length=100, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    combined_summary = ' '.join(summaries)

    
    summary_sentences = sent_tokenize(combined_summary)

   
    if num_sentences > len(summary_sentences):
        raise ValueError(f"The number of sentences requested ({num_sentences}) is greater than the number of sentences in the summary ({len(summary_sentences)}).")
    
   
    summary_output = ' '.join(summary_sentences[:num_sentences])  
    
    return summary_output
