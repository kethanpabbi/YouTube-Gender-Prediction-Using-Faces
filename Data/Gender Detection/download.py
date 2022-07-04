

from pytube import YouTube 
  
# where to save 
SAVE_PATH = "/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/" #to_do 
  
# link of the video to be downloaded 
link="https://www.youtube.com/watch?v=cB-DVomcEb4"
  
try: 
    # object creation using YouTube
    # which was imported in the beginning 
    yt = YouTube(link) 
except: 
    print("Connection Error") #to handle exception 

yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution')[-1].download()

print('Task Completed!') 