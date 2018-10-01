import cv2

from camera_interface.flir_api.flir_camera import FLIRCamera

flir = FLIRCamera(0)
flir_op = flir.get_out_pipe()
flir.run()

while True:
    flir_op.wait()
    frame = flir_op.pull()[1]
    cv2.imshow("Flir Camera",frame)
    cv2.waitKey(1)



