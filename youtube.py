import streamlit as st, torch
from datetime import date
from youtube_microservices import setup, fetch_data, process_data, analyse_data


def youtube():
    
    # Title & Icon
    _, col1, col2 = st.columns([2.2, 0.45, 3])

    with col1:
        st.image('resources/images/yt_bnw.png', width=100)
    with col2:
        st.markdown("<h1 style='text-align: left;'>YouTube</h1>", unsafe_allow_html=True)     
    
    # User inputs
    channel_name = st.text_input('Enter YouTube channel name')

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

    # Button Functionality
    if analyse:
        st.markdown('<br></br>', unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>Fetching Data</h3>", unsafe_allow_html=True)
        youtube = setup.execute()
        
        # Fetching data
        transcripts = fetch_data.execute(
            youtube=youtube,
            channel_name=channel_name, 
            start_date=start_date, 
            end_date=end_date
        )

        # Cleaning data
        st.markdown('<br></br>', unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>Cleaning Transcripts</h3>", unsafe_allow_html=True)
        cleaned_transcripts = process_data.execute(transcripts=transcripts)
        
        # Analysing data
        # overall_sentiment, sentiment_percentages, non_zero_word_count = analyse_data.execute(normalized_transcripts=normalized_transcripts)
        st.markdown('<br></br>', unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>Analysing Transcripts</h3>", unsafe_allow_html=True)
        sentiment_scores = analyse_data.execute(cleaned_transcripts)

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


        # Count the number of positive and negative sentiments
        positive_count = sum(score > 0.5 for score in sentiment_scores)
        negative_count = sum(score < 0.5 for score in sentiment_scores)
        

        data = {'Sentiment': ['Positive', 'Negative'], 'Count': [positive_count, negative_count]}

        # Create a bar chart using st.bar_chart
        st.bar_chart(data, x='Sentiment', y='Count', use_container_width=True)

 
# youtube()
    