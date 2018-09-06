import cv2
import os
import ffmpeg
#
# image_folder = 'MOT16/train/MOT16-02_1/img1'
# video_name = 'video.avi'
#
# images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
# frame = cv2.imread(os.path.join(image_folder, images[0]))
# height, width, layers = frame.shape
#
# video = cv2.VideoWriter(video_name, -1, 1, (width,height))
#
# for image in images:
#     video.write(cv2.imread(os.path.join(image_folder, image)))
#
# cv2.destroyAllWindows()
# video.release()
#


# os.system("ffmpeg -r 1 -i image_folder/%01d.jpg -vcodec mpeg4 -y movie.mp4")

import cv2
import os
import ffmpeg

#
#
image_folder = 'MOT16/train/MOT16-02_1/img1'
video_name = 'video.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, -1, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

