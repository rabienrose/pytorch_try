import torch
import lineNet
import gen_line
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np

pre_model="./chamo1.pth"
net = lineNet.Net()
net.load_state_dict(torch.load(pre_model))
net.eval()

img, target_py = gen_line.genAData()
img_gray = img.convert('L')
trans_toTensor = transforms.ToTensor()
input=trans_toTensor(img_gray)
input=input*2-1
input_batch=input.unsqueeze(0)
out = net(input_batch)
out_ny = out.detach().numpy()
out_ny=out_ny[0]
out_ny=-out_ny/out_ny[2]
a=out_ny[0]
c=out_ny[1]
k=a
x0=0
y0=c
p1_x=0
p1_y=k*(p1_x-x0)+y0
p2_x=1
p2_y=k*(p2_x-x0)+y0

draw = ImageDraw.Draw(img)
draw.line(((p1_x*gen_line.img_size, p1_y*gen_line.img_size) + (p2_x*gen_line.img_size,p2_y*gen_line.img_size)), fill="red", width=1)
img=img.resize((128,128), Image.ANTIALIAS)
img.show()