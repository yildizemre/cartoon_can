import uvicorn
from fastapi import FastAPI, File,Form,Body,status
from typing import List,Optional
import cv2
import numpy as np
import pytesseract
import os
import io
from fastapi.responses import JSONResponse
import datetime
hypegenai = FastAPI()
import numpy as np
import cv2
from PIL import ImageFile
import os
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
from model import Generator
import datetime
from starlette.responses import StreamingResponse
from fastapi.responses import FileResponse

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
    return out
@hypegenai.post("/api/file")
async def c_file(
    screen_image: bytes = File(...),
    ):
        try:
            jpg_as_np = np.frombuffer(screen_image, dtype=np.uint8)
            img_decode = cv2.imdecode(jpg_as_np, cv2.IMREAD_COLOR)
            image = img_decode.reshape(480,848,3)
            # print(image)
            datetime_str=datetime.datetime.now()
            cv2.imwrite("./samples/gonderilen_img/"+str(datetime_str)+".jpg",image)
            cv2.imwrite("./samples/image/"+str(datetime_str)+".jpg",image)
            image_path=str(datetime_str)+".jpg"

            test1=test(image_path)
            frame=cv2.imread("./static/img/"+str(image_path))
            frame=cv2.resize(frame,(848,480))
            data2 = cv2.imencode(".jpg", frame)[1]
            print(data2)
            content1 = {
                "image_path":str(data2.tobytes())
            }
            
            return JSONResponse(status_code=status.HTTP_200_OK, content=content1)

                # return str(content1)
        except:
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={'value':'file type error'})
      

if __name__ == "__main__":
	uvicorn.run(hypegenai)

    #usr/share/4.00/tessdata