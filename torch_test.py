import torch
import torchvision.transforms as T
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import cv2

if __name__ == "__main__":
    model = torch.load("torch.pkl")
    summary(model, (3, 650, 650))

    image_path = "test.jpg"
    img = PIL.Image.open(image_path)
    img = img.resize((650,650))
    with torch.no_grad():
        img_t = T.ToTensor()(img)
        mean = torch.as_tensor([0.4850, 0.4560, 0.4060])
        std = torch.as_tensor([0.2290, 0.2240, 0.2250])
        img_t = (img_t-mean[...,None,None]) / std[...,None,None]
        img_t = img_t[None]
        p,img_hr,b = model(img_t)[0]
        img_hr = img_hr*std[...,None,None] + mean[...,None,None] 
        img_hr_np = img_hr.to("cpu").numpy()
        img_hr_np = img_hr_np.transpose((1,2,0))

    fig,ax = plt.subplots(figsize=(9, 9))
    ax.axis('off')
    ax.imshow(img_hr_np, 'binary')
    ax.get_figure().savefig("res.jpg")

