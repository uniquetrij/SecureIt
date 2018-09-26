import cv2


class VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    def __init__(self, path, height, width, fps):
        self.__out = cv2.VideoWriter(path,VideoWriter.fourcc, fps, (height, width))


    def write(self, frame):
        self.__out.write(frame)

    def finish(self):
        self.__out.release()
