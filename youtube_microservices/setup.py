from googleapiclient.discovery import build

def execute():
    # YouTube Data API credentials
    #api_key = 'AIzaSyCmgwM6JEZ-KgeipnnU7glfNCZQ-1ezDz8' # Pratik
    # api_key = 'AIzaSyCafoXipft7GCX5UMQlDzRka0razjXHdVg' # Sthita
    api_key='AIzaSyDAsW-A7wQKd0K653nwSHRqEt6je4r6A-g' #Ujjwal

    # Create YouTube API client
    youtube = build('youtube', 'v3', developerKey=api_key)
    return youtube