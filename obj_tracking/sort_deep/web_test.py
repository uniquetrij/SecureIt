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
image_folder = '/home/developer/Desktop/folder/'

import cv2
import os

# fourcc = cv2.VideoWriter_fourcc(*'MPEG')
# out = cv2.VideoWriter('/home/developer/PycharmProjects/SecureIt/obj_tracking/sort_deep/MOT16/train/MOT16-02/test.avi', fourcc, 30, (width, height))

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
print(height, width, layers)

# video = cv2.VideoWriter_fourcc(video_name, -1, 1, (width,height))
fourcc = cv2.VideoWriter_fourcc(*'MPEG')
video = cv2.VideoWriter('/home/developer/Desktop/folder/test.mp4', fourcc, 25, (width, height))

images.sort()

for image in images:
    print(image)
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

