from bashmagic import *
import os
import glob
import json
os.chdir("/home/zoey/openpose")
print('a')

for i, path in enumerate(glob.iglob('/home/zoey/ucf_sports_actions/ucfaction/**/**/*.avi', recursive=True)):

    print(path)
    execute("./build/examples/openpose/openpose.bin --video {0} --write_keypoint_json /home/zoey/data/avi/ --no_display --render_pose 0".
            format(path))

    json_paths = get_files_paths_for_folder('/home/zoey/data/avi')
    for jpath in json_paths:
        dictionary = json.load(open(jpath))
        keypoint_list = dictionary['pose_keypoints']
        
	# 1. get pose data from dictionary
        # 2. put pose data into video list
        # 3. append video list to dataset list (similarly to our other script)
		
    execute('rm /home/zoey/data/avi/*')
