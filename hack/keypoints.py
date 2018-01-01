from bashmagic import *
import os
import glob
import json
import numpy as np
import moviepy.editor as mp
import cv2
import ntpath
from os.path import join

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

user = 'tim'

os.chdir("/home/{0}/openpose".format(user))
n = len(list(glob.iglob('/home/{0}/ucf_sports_actions/ucfaction/**/**/*.avi'.format(user), recursive=True)))
labels = []
data = []
nop = 1
Dist = 0
peopleID = -1

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



print(n, 'uden kek')
video_json_base_path = '/home/{0}/data/avi'.format(user)

for i, path in enumerate(glob.iglob('/home/{0}/ucf_sports_actions/ucfaction/**/**/*.avi'.format(user), recursive=True)):
    PointsList = []
    #a = mp.VideoFileClip(path)
    for key in label_action:
        if key in path:
            labels.append(label_action[key])
            break
    if i % 10 == 0:
        print(i/n)
    print(i, path)
    execute("./build/examples/openpose/openpose.bin --video {0} --write_keypoint_json /home/{1}/data/avi/ --no_display --render_pose 0".
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
        #print(keypoint_list)
        #if there is someone in the scene:
        Keypoints = np.array(keypoint_list)
        #Img[Keypoints[:, 0], Keypoints[:, 1]] = [255, 255, 255]
        ppl = int(Keypoints.shape[0]/18)
        for j in range(ppl-1):
            points = Keypoints[j*18:j*18+17,0:1]
            y1 = int(bbox(points).item(0))
            x1 = int(bbox(points).item(2))

            y2 = int(bbox(points).item(1))
            x2 = int(bbox(points).item(3))

            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if dist > Dist:
                Dist = dist
                peopleID = j
        #only get the largest bounding box of every frame
        PointsArray = Keypoints[peopleID*18:peopleID*18+17, :]
        PointsArr = np.array(PointsArray)
        print(PointsArr, path)
        if len(PointsArr): print(pts)
        PointsList.append(PointsArr.reshape(18, 3, 1, 1))
    print(np.array(PointsList).shape)
    data.append(np.concatenate(PointsList, 3))
    execute('rm /home/{0}/data/avi/*'.format(user))

l = np.max(list(map(lambda x: x.shape[3], data)))
output = np.zeros((n, 18, 3, 3, l), dtype=np.float32)

for i, points in enumerate(data):
    output[i, :, :, :, 0:points.shape[-1]] = points


np.save('/home/{0}/data/matrix.npy'.format(user), output)
np.save('/home/{0}/data/labels.npy'.format(user), np.array(labels))


