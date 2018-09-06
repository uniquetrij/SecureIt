from threading import Lock


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
        self.__lst = []
        self.__lock = Lock()

    def push(self, obj):
        self.__lock.acquire()
        self.__lst.append(obj)
        self.__lock.release()

    def pull(self, flush=False):
        try:
            self.__lock.acquire()
            if not self.is_closed():
                if len(self.__lst) > 0:
                    return True, self.__lst.pop(0)
                return False, None
            return False, None
        # else:
        #     raise Exception('I Dont Like Python!')
        finally:
            if flush:
                self.__lst.clear()
            self.__lock.release()




    def close(self):
        self.__lst.append(None)

    def is_closed(self):
        return len(self.__lst) == 1 and self.__lst[0] is None
