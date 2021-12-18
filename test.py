import os
import argparse

from PIL import Image
import numpy as np

import torch
from torchvision.transforms.functional import to_tensor, to_pil_image

from model import Generator


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


def test():
    device = "cpu"
    
    net = Generator()
    net.load_state_dict(torch.load('./weights/paprika.pt', map_location="cpu"))
    net.to(device).eval()
    print(f"model loaded: {'./weights/paprika.pt'}")
    
    os.makedirs('./samples/results', exist_ok=True)
    image_name="babam.jpg"
    
    if os.path.splitext(image_name)[-1].lower() not in [".jpg", ".png", ".bmp", ".tiff"]:
        pass
        
    image = load_image(os.path.join("./", image_name), False)

    with torch.no_grad():
        image = to_tensor(image).unsqueeze(0) * 2 - 1
        out = net(image.to(device), False).cpu()
        out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5
        out = to_pil_image(out)

    out.save(os.path.join('./samples/results', image_name))
    print(f"image saved: {image_name}")
    # import os
    # os.remove(image_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
   
    
    
    
    
    
    test()
