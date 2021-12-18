from flask import Flask,render_template, Response
import numpy as np
import mysql.connector
import io
import cv2
from PIL import ImageFile
from PIL import Image
from io import BytesIO
from PIL import Image
from PIL import ImageFile
import mysql.connector
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

from flask import Flask,render_template,flash,redirect,url_for,session,logging,request,jsonify
from wtforms import Form, StringField, PasswordField, validators,TextAreaField
from passlib.hash import sha256_crypt
import mysql.connector
from functools import wraps
import os
import glob
import os
import argparse
from pydantic.fields import T
import uvicorn
from fastapi import FastAPI, File,Form,Body,status
from typing import List,Optional
import cv2
import numpy as np
import pytesseract
import os
import io
from fastapi.responses import JSONResponse
# from pdf2image import convert_fkrom_path

from PIL import Image
import numpy as np

import torch
from torchvision.transforms.functional import to_tensor, to_pil_image

from model import Generator

import datetime
import time
UPLOAD_FOLDER = './'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ImageFile.LOAD_TRUNCATED_IMAGES = True    


torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def load_image(image_path, x32=False):
    img = Image.open(image_path).convert("RGB")

    if x32:
        def to_32s(x):
            return 256 if x < 256 else x - x % 32
        w, h = img.size
        img = img.resize((to_32s(w), to_32s(h)))

    return img


def test(image_name):
    device = "cpu"
    
    net = Generator()
    net.load_state_dict(torch.load('./weights/paprika.pt', map_location="cpu"))
    net.to(device).eval()
    print(f"model loaded: {'./weights/paprika.pt'}")
    
    os.makedirs('./samples/results', exist_ok=True)
    
    
    if os.path.splitext(image_name)[-1].lower() not in [".jpg", ".png", ".bmp", ".tiff"]:
        pass
        
    image = load_image(os.path.join("./samples/image/", image_name), False)

    with torch.no_grad():
        image = to_tensor(image).unsqueeze(0) * 2 - 1
        out = net(image.to(device), False).cpu()
        out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5
        out = to_pil_image(out)

    out.save(os.path.join('./static/img', image_name))
    print(f"image saved: {image_name}")
    return True
    # import os
    # os.remove(image_name)

@app.route('/api/file', methods=['POST'])
def eventPost():
    d = request.files.to_dict()
    content = request.form.to_dict()

    image=request.files['screen_image'].read()
    jpg_as_np = np.frombuffer(image, dtype=np.uint8)
    img_decode = cv2.imdecode(jpg_as_np, cv2.IMREAD_COLOR)
    image = img_decode.reshape(480,848,3)
    # cv2.imwrite("./samples/image/cartoon_image.jpg",image)

    datetime_str=datetime.datetime.now()
    cv2.imwrite("./samples/gonderilen_img/"+str(datetime_str)+".jpg",image)
    cv2.imwrite("./samples/image/"+str(datetime_str)+".jpg",image)
    image_path=str(datetime_str)+".jpg"
   
    
    
  
    test1=test(image_path)
    content1 = {
        "image_path":"./static/img/"+str(image_path)
    }
    if test1==True:

        return str(content1)
    else:
        return "FALSE"  
    # cv2.imshow("image",image)
    # cv2.waitKey(0)
    
if __name__ == '__main__':
    app.run( host="0.0.0.0" ,threaded=True,debug=True)