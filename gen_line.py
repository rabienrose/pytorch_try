from PIL import Image, ImageDraw
import random 
import math

img_size=32
i=0

def genAData(save_file=False):
    img = Image.new('RGB', (img_size, img_size), color = 'white')
    draw = ImageDraw.Draw(img)
    x0=random.random()
    y0=random.random()
    rad=random.uniform(-3.1415926, 3.1415926)
    k=math.tan(rad)
    p1_x=0
    p1_y=k*(p1_x-x0)+y0
    p2_x=1
    p2_y=k*(p2_x-x0)+y0
    r=random.randint(0, 255)
    g=random.randint(0, 255)
    b=random.randint(0, 255)
    w=int(random.random()*4)+1
    draw.line(((p1_x*img_size, p1_y*img_size) + (p2_x*img_size,p2_y*img_size)), fill=(r,g,b), width=w)
    for j in range(1, 100):
        x_noise=random.random()*img_size
        y_noise=random.random()*img_size
        ss=random.random()*3
        draw.ellipse([(x_noise, y_noise), (x_noise+ss, y_noise+ss)], fill='white')
    del draw
    if save_file:
        img_id=i+100000
        img = img.convert('LA')
        img.save('./data/line_'+str(img_id)+'.png')
        file1 = open('./data/line_'+str(img_id)+'.txt',"w")
        file1.write(str(x0)+','+str(y0)+','+str(rad))
        file1.close()

    target=[k, -k*x0+y0, -1]
    return img, target

