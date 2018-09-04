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
    def __init__(self):
        self.lst = []

    def push(self, obj):
        self.lst.append(obj)

    def pull(self):
        if not self.is_closed():
            if len(self.lst) > 0:
                return True, self.lst.pop(0)
            return False, None
        return False, None
        # else:
        #     raise Exception('I Dont Like Python!')


    def close(self):
        self.lst.append(None)

    def is_closed(self):
        return len(self.lst) == 1 and self.lst[0] is None
