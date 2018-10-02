import sys
# sys.path.insert(0,"agr")
from age_detection.age_detection import age_detection_new
import os
from moviepy.editor import VideoFileClip

age_detect = age_detection_new.AgeDetection()

age_detect.live_cv2()