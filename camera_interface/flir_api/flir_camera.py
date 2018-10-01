from threading import Thread

import  PySpin

from tf_session.tf_session_utils import Pipe

class BoolNode(PySpin.BooleanNode):
    
    def __init__(self,  *args, **kwargs):
        super(BoolNode, self).__init__( *args, **kwargs)
        # pass

    # def SetValue(self, Value, Verify=True):



class FLIRCamera(object):
    def __init__(self, index=None):
        if index is None:
            index = 0

        # PySpin.CameraPtr.AcquisitionFrameRateEnable
        self.__system = PySpin.System.GetInstance()
        self.__cam_list = self.__system.GetCameras()
        self.__cam = self.__cam_list.GetByIndex(index)
        self.__cam.Init()

        # print(type(self.__cam))
        self.__cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        # self.__Ibool = BoolNode()
        # print(self.__Ibool)
        # self.__Ibool.SetValue(Value=True)
        # self.__cam.AcquisitionFrameRateEnable(self.__Ibool)
        # self.__cam.AcquisitionFrameRate(25)

        self.__cam.BeginAcquisition()



        self.__out_pipe = Pipe(self.__out_pipe_process)
        self.__thread = None

    def __del__(self):
        self.close()

    def __out_pipe_process(self, image_primary):
        image_primary = image_primary.Convert(PySpin.PixelFormat_BGR8, PySpin.HQ_LINEAR)
        image_array = image_primary.GetNDArray()
        return image_array

    def close(self):
        self.__thread = None
        self.__cam.EndAcquisition()
        self.__cam.DeInit()
        del self.__cam
        del self.__cam_list
        del self.__system
        self.__out_pipe.close()

    def run(self):
        if self.__thread is None:
            self.__thread = Thread(target=self.__run)
            self.__thread.start()

    def __run(self):
        while self.__thread:
            try:
                self.__job(self.__cam.GetNextImage())
            except:
                self.close()

    def __job(self, image_primary):
        self.__out_pipe.push(image_primary)

    def get_out_pipe(self):
        return self.__out_pipe

