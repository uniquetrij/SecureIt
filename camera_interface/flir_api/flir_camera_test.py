import cv2
import time
from camera_interface.flir_api.flir_camera import FLIRCamera
import  numpy as np
flir = FLIRCamera(0)
flir_op = flir.get_out_pipe()
flir.run()
try:
    time1 = time.time()
    count = 0
    while True:
        flir_op.wait()
        frame = flir_op.pull()[1]
        # print(np.sum(frame[0] - frame[1]))
        count+=1
        cv2.imshow("Flir Camera",frame)
        cv2.waitKey(1)
except KeyboardInterrupt:
    pass
finally:
    time2 = time.time()
    diff = time2 - time1
    print(diff)
    print(count)


