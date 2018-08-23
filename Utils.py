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