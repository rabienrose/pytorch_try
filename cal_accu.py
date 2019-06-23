import torch
import torch.nn as nn
import lineNet
import torchvision.transforms as transforms
from numpy import linalg as LA
import numpy as np
import gen_line

pre_model="./chamo1.pth"
net = lineNet.Net()
net.load_state_dict(torch.load(pre_model))
net.eval()

batch_size=1000
input_list=[]
target_list=[]
for k in range(0, batch_size):
    img, target_py = gen_line.genAData()
    img = img.convert('L')
    trans_toTensor = transforms.ToTensor()
    input=trans_toTensor(img)
    input=input*2-1
    input_list.append(input)
    target_py=target_py/LA.norm(np.asarray(target_py))
    target = torch.FloatTensor(target_py)
    target_list.append(target)

input_batch = torch.stack(input_list)
target_batch = torch.stack(target_list)
out = net(input_batch)
criterion = nn.MSELoss()
loss = criterion(out, target_batch)
print(loss.tolist())