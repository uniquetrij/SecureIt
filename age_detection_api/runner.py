import sys
sys.path.insert(0,"agr")
from age_detection_api.age_detection import age_detection_new
import os
from moviepy.editor import VideoFileClip

class Runner:
	def __init__(self):
		self.__age_detect = age_detection_new.AgeDetection()

	def main(self, INPUT_FILE,subclip_start = None,subclip_end = None):
		age_detection_new.AgeDetection.frame=0
		age_detection_new.AgeDetection.max_frame=0
		age_detection_new.AgeDetection.progress=0
		INPUT_DIRECTORY = '../videos'
		OUTPUT_DIRECTORY = '../output'
		OUTPUT_FILE = 'processed.mp4'
		self.__age_detect.reset_data()
		vid_output = os.path.join(OUTPUT_DIRECTORY, OUTPUT_FILE)
		vid_input = os.path.join(INPUT_DIRECTORY, INPUT_FILE)
		clip = VideoFileClip(vid_input)#.subclip(subclip_start,subclip_end)
		age_detection_new.AgeDetection.max_frame = clip.duration * clip.fps
		vid = clip.fl_image(self.__age_detect.pipeline)
		self.__age_detect.create_metadata()
		vid.write_videofile(vid_output, audio=False)

	def progress(self):
		return age_detection_new.AgeDetection.progress


#main('age_detection_api.mp4')
