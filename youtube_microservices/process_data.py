import re

import streamlit as st
import tqdm
from tqdm import tqdm
# Cleaning data
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

# Driver function
def execute(transcripts):
    cleaned_transcripts = []
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    with tqdm(total=len(transcripts), bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        for i, transcript in enumerate(transcripts):
            cleaned_transcript = clean_text(transcript)
            cleaned_transcripts.append(cleaned_transcript)
            progress = (i + 1) / len(transcripts)
            progress_bar.progress(progress)
            pbar.update(1)

            progress = (i + 1) / len(transcripts)
            progress_bar.progress(progress)

            progress_percent = int(progress * 100)
            progress_text.text(f"Text Cleaning Progress: {progress_percent}%")
            pbar.update(1)
            i += 1

    return cleaned_transcripts