from threading import Thread

from tf_session.tf_session_utils import Pipe

pipe = Pipe(limit=None)

def writer():
    i = 0
    while True:
        if pipe.push(i):
            i+=1
        else:
            pipe.push_wait()

def reader():
    j=-1
    while True:
        ret, i = pipe.pull(True)
        if ret:
            print(i)
            j=i
        else:
            pipe.pull_wait()


Thread(target=writer).start()
Thread(target=reader).start()