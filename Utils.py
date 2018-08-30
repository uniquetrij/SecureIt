class VideoStreamer:
    hasNew = False
    newFrame = None

    def set(self, newFrame):
        self.newFrame = newFrame
        self.hasNew = True

    def get(self):
        if self.hasNew:
            return True, self.newFrame
        return False, None

class Pipe:
    __tag = None

    hasNew = False
    newFrame = None

    def __init__(self, tag=None):
        self.__tag = tag

    def push(self, newFrame):
        self.newFrame = newFrame
        self.hasNew = True

    def pull(self):
        if self.hasNew:
            return True, self.newFrame
        return False, None

    def getTag(self):
        return self.__tag






