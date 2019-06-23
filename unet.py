import torch
import torch.nn as nn
import lineNet
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import random 
import math
import torch.optim as optim
import time
from numpy import linalg as LA
import numpy as np
import gen_line

pre_model="./chamo.pth"
use_pretrain=True
device = torch.device("cuda:0")
net = lineNet.Net()
if use_pretrain:
    net.load_state_dict(torch.load(pre_model))
    net.eval()

net.to(device)
img_size=32
lr_chamo=0.01
optimizer = optim.SGD(net.parameters(), lr=lr_chamo)
input_list=[]
target_list=[]

sample_count=10000
batch_size=1000
# for k in range(0, sample_count):
#     img, target_py = gen_line.genAData()
#     img = img.convert('L')
#     trans_toTensor = transforms.ToTensor()
#     #trans_toPIL = transforms.ToPILImage()
#     input=trans_toTensor(img)
#     input=input*2-1
#     input_list.append(input)
#     #target_py=[x0*2-1, y0*2-1, rad/3.1415926]
#     target_py=target_py/LA.norm(np.asarray(target_py))
#     target = torch.FloatTensor(target_py)
#     target_list.append(target)
min_loss=100000
no_progress_count=0
for i in range(0, 100000):
    #number_list=list(range(0, sample_count))
    #random.shuffle(number_list)
    #input_list_batch=[]
    #target_list_batch = []
    input_list = []
    target_list = []
    for j in range(0,batch_size):
        img, target_py = gen_line.genAData()
        img = img.convert('L')
        trans_toTensor = transforms.ToTensor()
        # trans_toPIL = transforms.ToPILImage()
        input = trans_toTensor(img)
        input = input * 2 - 1
        input_list.append(input)
        # target_py=[x0*2-1, y0*2-1, rad/3.1415926]
        target_py = target_py / LA.norm(np.asarray(target_py))
        target = torch.FloatTensor(target_py)
        target_list.append(target)
        #input_list_batch.append(input_list[j])
        #target_list_batch.append(target_list[j])
    input_batch = torch.stack(input_list)
    target_batch = torch.stack(target_list)
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    start_time = time.time()
    optimizer.zero_grad()
    out = net(input_batch)
    criterion = nn.MSELoss()
    loss = criterion(out, target_batch)
    if min_loss>loss.tolist():
        min_loss=loss.tolist()
        torch.save(net.state_dict(), pre_model)
        print(loss.tolist())
        no_progress_count=0
    else:
        no_progress_count=no_progress_count+1
    if no_progress_count>100:
        no_progress_count=0
        lr_chamo = lr_chamo * 0.9
        print("new learning rate: "+str(lr_chamo))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_chamo
    loss.backward()
    optimizer.step() 
    #print(time.time() - start_time)
    

