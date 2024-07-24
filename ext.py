import spacy
from collections import Counter
from heapq import nlargest

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
   
    doc = nlp(text)
    tokens = [
        token.text.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and token.text != "\n"
    ]
    return doc, tokens

def extractive_summarize(text, num_sentences=4):
   
    doc, tokens = preprocess_text(text)

   
    word_freq = Counter(tokens)
    max_freq = max(word_freq.values())

   
    for word in word_freq.keys():
        word_freq[word] = word_freq[word] / max_freq

    sent_token = [sent.text for sent in doc.sents]
    sent_score = {}
    for sent in sent_token:
        for word in sent.split():
            if word.lower() in word_freq.keys():
                if sent not in sent_score.keys():
                    sent_score[sent] = word_freq[word]
                else:
                    sent_score[sent] += word_freq[word]

  
    if num_sentences > len(sent_token):
        raise ValueError(f"Number of sentences requested ({num_sentences}) is greater than the number of sentences in the input text ({len(sent_token)}).")
    
    top_sentences = nlargest(num_sentences, sent_score, key=sent_score.get)
    summary = " ".join(top_sentences)
    
    if len(top_sentences) < num_sentences:
        raise ValueError(f"Number of sentences in the summary ({len(top_sentences)}) is less than the requested number of sentences ({num_sentences}).")

    return summary
