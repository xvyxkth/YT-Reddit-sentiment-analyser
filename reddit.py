import streamlit as st
from datetime import date
import datetime, asyncpraw, asyncio, re, torch
from transformers import  RobertaForSequenceClassification, RobertaTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

def reddit():

    # Title & Icon
    _, col1, col2 = st.columns([2.3, 0.45, 3])

    with col1:
        st.image('resources/images/reddit_white.png', width=100)
    with col2:
        st.markdown("<h1 style='text-align: left;'>Reddit</h1>", unsafe_allow_html=True)     

    # User Input
    subreddit_name = st.text_input('Enter sub reddit name')

    c1, c2 = st.columns([1, 1])
    with c1:
        start_date = st.date_input('Select start date')
    with c2:
        end_date = st.date_input('Select end date')

    today = date.today()
    if(start_date > end_date or end_date > today):
        st.warning('Please select valid dates')
    
    # Button
    analyse = False
    _, cl2, _ = st.columns([2.65,1,2])
    with cl2:
        analyse = st.button('Analyse')

    # Button functionality
    if analyse:
        # Function to clean data
        def clean_text(text):
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            text = text.lower()            
            return text
        
        # Fetching data from subreddit
        async def main():
            async with asyncpraw.Reddit(client_id="WQK5n-GHzXU5IihE1pIH7Q",
                                    client_secret="CU6HY5cmhFBTf6X_sID-MTCgkokjXg",
                                    user_agent="Sentimental Analysis") as reddit:

                subreddit = await reddit.subreddit(subreddit_name)

                start_time = datetime.datetime.combine(start_date, datetime.datetime.min.time())

                end_time = datetime.datetime.combine(end_date, datetime.datetime.max.time())

                posts = subreddit.new(limit=None)

                post_contents = []

                st.markdown('<br></br>', unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: center;'>Post Titles</h3>", unsafe_allow_html=True)

                async for post in posts:
                    if start_time <= datetime.datetime.fromtimestamp(post.created_utc) <= end_time:
                        post_contents.append(post.title + " " + post.selftext)
                        
                        st.text(f"Title: {post.title}\n")


                cleaned_contents = []

                # Cleaning data
                st.markdown('<br></br>', unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: center;'>Cleaning Text</h3>", unsafe_allow_html=True)
                

                # Display the progress bar for text cleaning
                progress_bar = st.progress(0)
                progress_text = st.empty()

                with tqdm(total=len(post_contents), desc="Cleaning Text") as pbar:
                    for i, text in enumerate(post_contents):
                        cleaned_text = clean_text(text)
                        cleaned_contents.append(cleaned_text)

                        # Update the progress bar for text cleaning
                        progress = (i + 1) / len(post_contents)
                        progress_bar.progress(progress)

                        # Update the progress text for text cleaning
                        progress_percent = int(progress * 100)
                        progress_text.text(f"Cleaning Text: {progress_percent}%")
                        pbar.update(1)

                
                await perform_sentiment_analysis(cleaned_contents)

                # Analysing data using RoBERTa
        async def perform_sentiment_analysis(cleaned_contents):
            
            st.markdown('<br></br>', unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center;'>Analysing Text</h3>", unsafe_allow_html=True)
            model = RobertaForSequenceClassification.from_pretrained('roberta-base')
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            sentiment_scores = []

            # Display the progress bar
            progress_bar = st.progress(0)
            progress_text = st.empty()

            with tqdm(total=len(cleaned_contents), desc="Analysing Text") as pbar:
                for i, text in enumerate(cleaned_contents):
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
                    #print(len(chunk_scores))

                    sentiment_scores.append(average_score)

                    # Update the progress bar
                    progress = (i + 1) / len(cleaned_contents)
                    progress_bar.progress(progress)

                    # Update the progress text
                    progress_percent = int(progress * 100)
                    progress_text.text(f"Analysing Text: {progress_percent}%")
                    pbar.update(1)
                
                
            for i, text in enumerate(cleaned_contents):
                sentiment_score = sentiment_scores[i]
                sentiment_label = "Positive" if sentiment_score >= 0.5 else "Negative"

            st.markdown('<br></br>', unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center;'>Results</h3>", unsafe_allow_html=True)
            mean_sentiment_score = torch.mean(torch.tensor(sentiment_scores))
            overall_sentiment = ''
            if mean_sentiment_score >= 0.5:
                overall_sentiment = 'POSITIVE'
            else:
                overall_sentiment = 'NEGATIVE'

            st.markdown(f"<h4 style='text-align: left;'>Mean Sentiment Score: {mean_sentiment_score.item()}</h4>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='text-align: left;'>Mean Sentiment: {overall_sentiment}</h4>", unsafe_allow_html=True)


            positive_count = sum(score > 0.5 for score in sentiment_scores)
            negative_count = sum(score < 0.5 for score in sentiment_scores)

            data = {'Sentiment': ['Positive', 'Negative'], 'Count': [positive_count, negative_count]}
            df = pd.DataFrame(data)  # Create a DataFrame from the data dictionary

            # Create a bar chart using st.bar_chart
            st.bar_chart(df, x='Sentiment', y='Count', use_container_width=True)
            
            
        asyncio.run(main())         
 

# reddit()