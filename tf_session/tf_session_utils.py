import threading
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
    def __init__(self, process=None):
        self.__lst = []
        self.__lock = Lock()
        self.__closed = False
        self.__process = process
        self.__pause_resume = threading.Event()
        self.__joint = []

    def push(self, obj):
        if self.is_closed():
            raise Exception("can't push into a closed pipe")

        if self.__process:
            obj = self.__process(obj)

        self.__lock.acquire()
        try:
            self.__lst.append(obj)
            self.__pause_resume.set()
            for pipe in self.__joint:
                pipe.push(obj)
        finally:
            self.__lock.release()

    def pull(self, flush=False):
        if self.is_closed():
            raise Exception("can't pull from a closed pipe")

        self.__lock.acquire()
        try:
            if not self.is_closed():
                if len(self.__lst) > 0:
                    if flush:
                        return True, self.__lst.pop(0)
                    else:
                        return True, self.__lst.pop(0)
                return False, None
            return False, None
        # else:
        #     raise Exception('I Dont Like Python!')
        finally:
            if flush:
                self.__lst.clear()

            if not self.__lst:
                self.__pause_resume.clear()
            self.__lock.release()

    def close(self):
        self.__closed = True

    def is_closed(self):
        return len(self.__lst) == 0 and self.__closed == True

    def wait(self):
        self.__pause_resume.wait()

    def join(self, pipe):
        self.__joint.append(pipe)


