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

sample_count=100000