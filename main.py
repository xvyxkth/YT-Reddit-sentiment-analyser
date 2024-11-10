import streamlit as st
from reddit import reddit
from youtube import youtube

# def main():
# Settting page configurations
st.set_page_config(page_title='Sentiment Analyser', layout='wide')

# Title
st.markdown("<h1 style='text-align: center;'>Welcome to Sentiment Analyser!</h1>", unsafe_allow_html=True)

# Dropdown selection
app_selection = st.selectbox('Select Platform', ['Select', 'Reddit', 'YouTube'])

# 
if app_selection == 'Select':
    pass
elif app_selection == 'Reddit':
    st.markdown('<br></br>', unsafe_allow_html=True)
    reddit()
elif app_selection == 'YouTube':
    st.markdown('<br></br>', unsafe_allow_html=True)    
    youtube()

# main()