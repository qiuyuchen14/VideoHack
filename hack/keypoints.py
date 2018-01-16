from bashmagic import *
import os
import glob
import json
import numpy as np
import moviepy.editor as mp
import cv2
import ntpath
from os.path import join
#############################################
#To Do:
#predict bounding box region
#predict each keypoint position in the next frames
#given a catogory, produce series of actions through learning: This is very interesting to me because you can
#personalize your robot to do different tasks. It can be possibly associated with reinforcement learning
# so that the robot can be initialized with certain tasks and then learn how to finish it in complex environments
#
#############################################
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
def nvector(point1x, point1y, point2x, point2y):
    newx = (point2x - point1x)
    newy = (point2y - point1y)
    a = newy/newx
    return a


user = 'zoey'

os.chdir("/home/{0}/openpose".format(user))
#n = len(list(glob.iglob('/home/{0}/ucf_sports_actions/ucfaction/*/*/*.avi'.format(user), recursive=True)))
n = len(list(glob.iglob('/home/{0}/ucf_sports_actions/ucfaction/*/*/*.avi'.format(user), recursive=True)))
labels = []
data = []
data1 = []
nop = 1


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

#label_action['basketball'] = 0
#label_action['golf_swing'] = 1
#label_action['soccer_juggling'] = 2
#label_action['trampoline_jumping'] = 3
#label_action['biking'] = 4
#label_action['horse_riding'] = 5
#label_action['swing'] = 6
#label_action['volleyball_spiking'] = 7
#label_action['diving'] = 8
#label_action['tennis_swing'] = 9
#label_action['walking'] = 10



video_json_base_path = '/home/{0}/data/avi'.format(user)

for i, path in enumerate(glob.iglob('/home/{0}/ucf_sports_actions/ucfaction/*/*/*.avi'.format(user), recursive=True)):
    PointsList = []
    PointRelation = []

    #a = mp.VideoFileClip(path)
    for key in label_action:
        if key in path:
            labels.append(label_action[key])
            break
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
            #alternative: see relative movements among keypoints
            PointRel = np.zeros((18, 18))
            for r in range(18):
                for c in range(18):
                    PointReL[r][c] = nvector(PointsArr[r][0],PointsArr[r][1], PointsArr[c][0], PointsArr[c][1])

            PointsList.append(PointsArr.reshape(18, 2, 1, 1))# number of keypoints, x & y, number of people, one frame
            PointRelation.append(PointReL.reshape(18, 18, 1, 1)) # number of keypoints, number of keypoitns, number of people, one frame
    data.append(np.concatenate(PointsList, 3))
    data1.append(np.concatenate(PointRelation, 3))
    execute('rm /home/{0}/data/avi/*'.format(user))

l = np.max(list(map(lambda x: x.shape[3], data)))
output = np.zeros((n, 18, 2, 1, l), dtype=np.float32)
output1 = np.zeros((n, 18, 18, l), dtype=np.float32)

for i, points in enumerate(data):
    output[i, :, :, :, 0:points.shape[-1]] = points

for i, relations in enumerate(data1):
    output1[i, :, :, :, 0:points.shape[-1]] = relations


np.save('/home/{0}/data/UCF/keypoints.npy'.format(user), output)
np.save('/home/{0}/data/UCF/ReLMovement.npy'.format(user), output1)
np.save('/home/{0}/data/UCF/labels.npy'.format(user), np.array(labels))


