

# from pytube import YouTube 
  
# # where to save 
# SAVE_PATH = "/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/video" #to_do 
  
# # link of the video to be downloaded 
# link="https://www.youtube.com/watch?v=BzLO2OKt3OU"
  
# try: 
#     # object creation using YouTube
#     # which was imported in the beginning 
#     yt = YouTube(link) 
# except: 
#     print("Connection Error") #to handle exception 

# #yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution')[-1].download()

# print('Task Completed!') 
# s = "144p"
# #yt.streams.filter(progressive=True, file_extension='mp4', resolution=s).download()
# #yt.streams.first().download()
# yt.streams.get_by_resolution()

# ./pytube_demo.py

from pytube import YouTube

def get_stream_for_res(streams, res):
    stream = list(filter(lambda x: x.resolution == res, streams))
    return stream
   
video_url = input("Enter YouTube Video URL: ").strip()
youtube_obj = YouTube(video_url)

video_res = input(f"Enter YouTube Video Resolution for {youtube_obj.title}: ").strip()
req_stream_obj = get_stream_for_res(youtube_obj.streams.filter(progressive = True, file_extension = 'mp4'), video_res)[0]

req_stream_obj.download()
print(f"YouTube Video {youtube_obj.title} Downloaded With Resolution {video_res}")