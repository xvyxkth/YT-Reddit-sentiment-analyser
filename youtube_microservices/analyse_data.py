from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import  RobertaForSequenceClassification, RobertaTokenizer
import torch
import streamlit as st
import tqdm
from tqdm import tqdm


def analyse(cleaned_transcripts):
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    sentiment_scores = []
    progress_bar = st.progress(0)
    progress_text = st.empty()

    with tqdm(total=len(cleaned_transcripts), bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        for i, text in enumerate(cleaned_transcripts):
            text_chunks = [text[i:i + 512] for i in range(0, len(text), 512)]
            chunk_scores = []

            for chunk in text_chunks:
                inputs = tokenizer(chunk, truncation=True, padding=True, return_tensors="pt").to(device)
                with torch.inference_mode():
                    model_output = model(**inputs)
            logits = model_output.logits
            probabilities = torch.softmax(logits, dim=1)
            sentiment_score = probabilities[:, 1].item()  # positive sentiment
            chunk_scores.append(sentiment_score)

            average_score = sum(chunk_scores) / len(chunk_scores)
            sentiment_scores.append(average_score)
            # Update the progress bar
            progress = (i + 1) / len(cleaned_transcripts)
            progress_bar.progress(progress)

            progress_percent = int(progress * 100)
            progress_text.text(f"Sentimental Analysis Progress: {progress_percent}%")
            pbar.update(1)

            

        
    for i, text in enumerate(cleaned_transcripts):
        sentiment_score = sentiment_scores[i]
        sentiment_label = "Positive" if sentiment_score >= 0.5 else "Negative"

    mean_sentiment_score = torch.mean(torch.tensor(sentiment_scores))
    overall_sentiment = ''
    if mean_sentiment_score >= 0.5:
        overall_sentiment = 'POSITIVE'
    else:
        overall_sentiment = 'NEGATIVE'
    
    print(sentiment_scores)
    return sentiment_scores


def execute(cleaned_transcripts):
    sentiment_scores = analyse(cleaned_transcripts)
    return sentiment_scores

    