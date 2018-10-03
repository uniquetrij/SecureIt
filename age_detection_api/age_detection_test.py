import sys

import os
# from moviepy.editor import VideoFileClip
from age_detection_api.age_detection.age_detection_new import AgeDetection

age_detect = AgeDetection()

age_detect.live_cv2()