import torch
import numpy as np
import torch.utils.data as tor
import time
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable

from bashmagic import *
import os
import glob
import json
import moviepy.editor as mp
import cv2
import ntpath
from os.path import join


model = torch.load('/home/zoey/data/model/model.pkl')
test = # I need to find some good videos

#get the biggest boundingbox:
def bbox(points):
    """
    [xmin xmax]
    [ymin ymax]
    """
    a = np.zeros((2, 2))
    a[:, 0] = np.min(points, axis=0)
    a[:, 1] = np.max(points, axis=0)
    return a

user = 'zoey'

os.chdir("/home/{0}/openpose".format(user))
n = len(list(glob.iglob('/home/{0}/action_youtube_naudio/diving/**/*.avi'.format(user), recursive=True)))
data = []
nop = 1

video_json_base_path = '/home/{0}/data/testdataset/avi'.format(user)

for i, path in enumerate(glob.iglob('/home/{0}/action_youtube_naudio/diving/diving/*.avi'.format(user), recursive=True)):
    PointsList = []
    if i % 10 == 0:
        print(i/n)
    print(i, path)
    execute('./build/examples/openpose/openpose.bin --video "{0}" --write_keypoint_json /home/{1}/data/avi/ --no_display --render_pose 0'.
            format(path, user))
    video_name = ntpath.basename(path)
    video = []
    #Img = np.zeros((a.shape[0], a.shape[1], 3), dtype=np.uint8)
    for i in range(10000):
        jpath = join(video_json_base_path, video_name.replace('.avi', '_') + str(i).zfill(12) + '_keypoints.json')
        if not os.path.exists(jpath): break

        dictionary = json.load(open(jpath))
        people = dictionary['people']
        keypoint_list = []

        for person in people:
            pts = person['pose_keypoints']
            if len(pts) > 3:
                keypoint_list.append(np.array(person['pose_keypoints']).reshape(18, 3))
        #if there is someone in the scene:
        if len(keypoint_list) > 0:
            Area = 0
            peopleID = -1
            Keypoints = np.array(keypoint_list)
            j = 0 #people id labeling
            for keypoints in Keypoints:
                points = []
                for pose in keypoints:
                    pose = np.array(pose)

                    if pose[2] > 0:
                        points.append(pose[0:2])
                points = np.array(points)
                y1 = int(bbox(points).item(0))
                x1 = int(bbox(points).item(2))

                y2 = int(bbox(points).item(1))
                x2 = int(bbox(points).item(3))

                area = np.abs(x2 - x1)*np.abs(y2 - y1)
                if area > Area:
                    Area = area
                    peopleID = j
                j = j+1
            #only get the largest bounding box of every frame
            PointsArray = Keypoints[peopleID, :, 0:2]
            PointsArr = np.array(PointsArray)
            PointsList.append(PointsArr.reshape(18, 2, 1, 1))# number of keypoints, x & y, number of people, one frame

    data.append(np.concatenate(PointsList, 3))
    execute('rm /home/{0}/data/testdataset/avi/*'.format(user))

l = np.max(list(map(lambda x: x.shape[3], data)))
output = np.zeros((n, 18, 2, 1, l), dtype=np.float32)

for i, points in enumerate(data):
    output[i, :, :, :, 0:points.shape[-1]] = points


np.save('/home/{0}/data/testdataset/matrix.npy'.format(user), output)

for epoch in range(40):
    accuracy = []
    model.train(False)
    for batch, label_val in vl:
        batch = Variable(batch)
        label_val = Variable(label_val)
        pred_val = model.forward(batch)
        loss = model.loss(pred_val, label_val)

        maxvalue, argmax = torch.topk(pred_val, 1)
        correct = torch.sum(argmax.data == label_val.view(-1, 1).data)
        accuracy.append(correct/batch.size(0))
    acc = np.mean(accuracy)
    print('test:', np.mean(accuracy))
