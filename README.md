# video_maker
util for encoding list of images into video
!pip install imageio

!pip install moviepy

### 5. 인퍼런스 랜더링
import glob
import os

import imageio
from moviepy.editor import VideoFileClip

import numpy as np
import cv2
import matplotlib.pyplot as plt

imageio.plugins.ffmpeg.download()
%matplotlib inline


#data_path = os.path.join(home, data, tag)
input_path = '/home/ubuntu/the_shell.mp4' #we inquire the backend-shell mp4 for stable encoding 
output_path = '/home/ubuntu/the_inferrred.mp4'
image_dir = '/home/ubuntu/data/ipx_2/src_r_512/image/'
infer_dir = './images/'

dir_list = [image_dir, infer_dir]
tgt_size = 512

target_resolution = (tgt_size, tgt_size * len(dir_list))

[print("%s : %d"%(d, len(glob.glob(d + '/*')))) for d in dir_list]

def process_video(input_img):
    global frames
    
    image_name = '%06d.png'%(frames)
    jpg_name = '%06d_g1.jpg'%frames
    image_names = [image_name, jpg_name]
    
    def get_img(path, size) :
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        return img

    img_list = []
    for d in dir_list :
        for image_name in image_names :
            path = os.path.join(d, image_name)

            if os.path.exists(path) :
                img = get_img(path, (tgt_size, tgt_size))
                img_list.append(img)

    if len(img_list) == len(dir_list) :
        input_img = np.concatenate(img_list, axis=1)        
    
    frames += 1
    
    return input_img

global frames
global prev
prev = None
frames = 0

clip1 = VideoFileClip(input_path, target_resolution=target_resolution)
clip = clip1.fl_image(process_video)#.subclip(0,25) #NOTE: this function expects color images!!
%time clip.write_videofile(output_path, audio=True)
