import threading
from threading import Lock
from time import time


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
    def __init__(self, process=None, limit=1):
        self.__lst = []
        self.__lock = Lock()
        self.__closed = False
        self.__process = process
        self.__pullable = threading.Event()
        self.__pushable = threading.Event()
        self.__joint = []
        self.__limit = limit
        self.__pushable.set()

    def push(self, obj):
        if self.is_closed():
            raise Exception("can't push into a closed pipe")

        if self.is_full():
            self.__lock.acquire()
            try:
                self.__pushable.clear()
                return False
            finally:
                self.__lock.release()

        if self.__process:
            obj = self.__process(obj)

        if obj is '\0':
            return

        self.__lock.acquire()
        try:
            self.__lst.append(obj)
            self.__pullable.set()
            for pipe in self.__joint:
                pipe.push(obj)
            return True
        except:
            return False
        finally:
            self.__lock.release()

    def pull(self, flush=False):
        if self.is_closed():
            raise Exception("can't pull from a closed pipe")

        self.__lock.acquire()
        try:
            if not self.is_closed():
                if len(self.__lst) > 0:
                    ret = self.__lst.pop(0)
                    self.__pushable.set()
                    return True, ret
                return False, None
            return False, None
        # else:
        #     raise Exception('I Dont Like Python!')
        finally:
            if flush:
                self.__lst.clear()

            if not self.__lst:
                self.__pullable.clear()
            self.__lock.release()

    def close(self):
        self.__closed = True
        self.__pullable.set()

    def is_closed(self):
        return len(self.__lst) == 0 and self.__closed == True

    def pull_wait(self, timeout=None):
        self.__pullable.wait(timeout)

    def push_wait(self, timeout=None):
        self.__pushable.wait(timeout)

    def join(self, pipe):
        self.__joint.append(pipe)

    def is_full(self):
        if not self.__limit:
            return False
        return len(self.__lst) >= self.__limit

    def flush(self):
        self.__lock.acquire()
        try:
            self.__lst.clear()
            if not self.__lst:
                self.__pullable.clear()
        finally:
            self.__lock.release()


class Inference:
    def __init__(self, input, return_pipe=None, meta_dict=None):
        self.__input = input
        self.__meta_dict = meta_dict
        if not self.__meta_dict:
            self.__meta_dict = {}

        self.__return_pipe = return_pipe
        self.__data = None
        self.__result = None
        self.ts = time()
        self.seq = None

    def get_input(self):
        return self.__input

    def get_meta_dict(self):
        return self.__meta_dict

    def get_return_pipe(self):
        return self.__return_pipe

    def set_result(self, result):
        self.__result = result
        if self.__return_pipe:
            self.__return_pipe.push(self)

    def get_result(self):
        return self.__result

    def set_data(self, data):
        self.__data = data

    def get_data(self):
        return self.__data

    def set_meta(self, key, val):
        if key not in self.__meta_dict.keys():
            self.__meta_dict[key] = val
        else:
            raise Exception

    def set_meta_force(self, key, val):
        self.__meta_dict[key] = val

    def get_meta(self, key):
        if key in self.__meta_dict.keys():
            return self.__meta_dict[key]
        return None

    def get_meta_or_default(self, key, val):
        if key in self.__meta_dict.keys():
            return self.__meta_dict[key]
        return val
