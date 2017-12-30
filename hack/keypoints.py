from bashmagic import *
import os
os.chdir("/home/zoey/openpose")


execute("./build/examples/openpose/openpose.bin --video examples/media/video.avi --write_keypoint_json output/ --no_display --render_pose 0")

