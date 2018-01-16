import torch
import scipy
import numpy as np
import imageio
import glob
import moviepy.editor as mp

from scipy.misc import imread

data = []

label_action = {}

label_action['basketball'] = 0
label_action['golf_swing'] = 1
label_action['soccer_juggling'] = 2
label_action['trampoline_jumping'] = 3
label_action['biking'] = 4
label_action['horse_riding'] = 5
label_action['swing'] = 6
label_action['volleyball_spiking'] = 7
label_action['diving'] = 8
label_action['tennis_swing'] = 9
label_action['walking'] = 10

labels = []
size = 32
n = len(list(glob.iglob('/home/zoey/action_youtube_naudio/*/*/*.avi', recursive=True)))
for i, filename in enumerate(glob.iglob('/home/zoey/action_youtube_naudio/*/*/*.avi', recursive=True)):
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



np.save('/home/zoey/data/Youtube/WholeImage/matrix.npy', output)
np.save('/home/zoey/data/Youtube/WholeImage/labels.npy', np.array(labels))



