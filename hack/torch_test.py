import torch
import scipy
import numpy as np
import imageio
import glob
import moviepy.editor as mp

from scipy.misc import imread

data = []

label_action = {}
label_action['Diving-Side'] = 0
label_action['Golf-Swing-Back'] = 1
label_action['Golf-Swing-Front'] = 2
label_action['Golf-Swing-Side'] = 3
label_action['Kicking-Front'] = 4
label_action['Kicking-Side'] = 5
label_action['Lifting'] = 6
label_action['Riding-Horse'] = 7
label_action['Run-Side'] = 8
label_action['SkateBoarding-Front'] = 9
label_action['Swing-Bench'] = 10
label_action['Swing-SideAngle'] = 11
label_action['Walk-Front'] = 12


labels = []
size = 32
n = len(list(glob.iglob('/home/zoey/ucf_sports_actions/ucfaction/*/*/*.avi', recursive=True)))
for i, filename in enumerate(glob.iglob('/home/zoey/ucf_sports_actions/ucfaction/*/*/*.avi', recursive=True)):
    a = mp.VideoFileClip(filename)
    print(a)
    for key in label_action:
        if key in filename:
            labels.append(label_action[key])
            break

    clip_resized = a.resize((size, size))
    if i % 10 == 0:
        print(i/n)

    video = []
    for img in clip_resized.iter_frames():
        video.append(img.reshape(size, size, 3, 1))
    data.append(np.concatenate(video, 3))

l = np.max(list(map(lambda x: x.shape[3], data)))
output = np.zeros((n, size, size, 3, l), dtype=np.float32)
for i, video in enumerate(data):
    output[i, :, :, :, 0:video.shape[-1]] = video

print(len(labels), n)



np.save('/home/zoey/data/matrix.npy', output)
np.save('/home/zoey/data/labels.npy', np.array(labels))



