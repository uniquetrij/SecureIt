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

        self.__system = PySpin.System.GetInstance()
        self.__cam_list = self.__system.GetCameras()
        self.__cam = self.__cam_list.GetByIndex(index)
        # self.__pixel_format = self.__cam.PixelFormat
        # print(self.__pixel_format.GetCurrentEntry())
        # self.__pixel_format = PySpin.PixelFormat_BGR8
        # print(self.__pixel_format)
        # self.__access_mode = self.__cam.PixelFormat.GetAccessMode
        self.__cam.Init()
        self.__nodemap = self.__cam.GetNodeMap()
        self.configure_custom_image_settings(self.__nodemap)
        # self.__cam.DeInit()
        # self.__cam.Init()
        self.__cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        print(self.__cam.EncoderMode)

        self.__node_acquisition_framerate = PySpin.CFloatPtr(self.__nodemap.GetNode('AcquisitionFrameRate'))
        self.__framerate_to_set = self.__node_acquisition_framerate.GetValue()

        print(self.__framerate_to_set)


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

    def configure_custom_image_settings(self, nodemap):
        """
        Configures a number of settings on the camera including offsets  X and Y, width,
        height, and pixel format. These settings must be applied before BeginAcquisition()
        is called; otherwise, they will be read only. Also, it is important to note that
        settings are applied immediately. This means if you plan to reduce the width and
        move the x offset accordingly, you need to apply such changes in the appropriate order.

        :param nodemap: GenICam nodemap.
        :type nodemap: INodeMap
        :return: True if successful, False otherwise.
        :rtype: bool
        """
        try:
            result = True

            # Apply mono 8 pixel format
            #
            # *** NOTES ***
            # Enumeration nodes are slightly more complicated to set than other
            # nodes. This is because setting an enumeration node requires working
            # with two nodes instead of the usual one.
            #
            # As such, there are a number of steps to setting an enumeration node:
            # retrieve the enumeration node from the nodemap, retrieve the desired
            # entry node from the enumeration node, retrieve the integer value from
            # the entry node, and set the new value of the enumeration node with
            # the integer value from the entry node.
            #
            # Retrieve the enumeration node from the nodemap
            node_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))

            if PySpin.IsAvailable(node_pixel_format) and PySpin.IsWritable(node_pixel_format):

                # Retrieve the desired entry node from the enumeration node
                node_pixel_format_mono8 = PySpin.CEnumEntryPtr(node_pixel_format.GetEntryByName('RGB8'))
                print(node_pixel_format.GetEntries())
                if PySpin.IsAvailable(node_pixel_format_mono8) and PySpin.IsReadable(node_pixel_format_mono8):

                    # Retrieve the integer value from the entry node
                    pixel_format_mono8 = node_pixel_format_mono8.GetValue()

                    # Set integer as new value for enumeration node
                    node_pixel_format.SetIntValue(pixel_format_mono8)

                    print('Pixel format set to %s...' % node_pixel_format.GetCurrentEntry().GetSymbolic())

                else:
                    print('Pixel format mono 8 not available...')

            else:
                print('Pixel format not available...')

            # Apply minimum to offset X
            #
            # *** NOTES ***
            # Numeric nodes have both a minimum and maximum. A minimum is retrieved
            # with the method GetMin(). Sometimes it can be important to check
            # minimums to ensure that your desired value is within range.
            node_offset_x = PySpin.CIntegerPtr(nodemap.GetNode('OffsetX'))
            if PySpin.IsAvailable(node_offset_x) and PySpin.IsWritable(node_offset_x):

                node_offset_x.SetValue(node_offset_x.GetMin())
                print('Offset X set to %i...' % node_offset_x.GetMin())

            else:
                print('Offset X not available...')

            # Apply minimum to offset Y
            #
            # *** NOTES ***
            # It is often desirable to check the increment as well. The increment
            # is a number of which a desired value must be a multiple of. Certain
            # nodes, such as those corresponding to offsets X and Y, have an
            # increment of 1, which basically means that any value within range
            # is appropriate. The increment is retrieved with the method GetInc().
            node_offset_y = PySpin.CIntegerPtr(nodemap.GetNode('OffsetY'))
            if PySpin.IsAvailable(node_offset_y) and PySpin.IsWritable(node_offset_y):

                node_offset_y.SetValue(node_offset_y.GetMin())
                print('Offset Y set to %i...' % node_offset_y.GetMin())

            else:
                print('Offset Y not available...')

            # Set maximum width
            #
            # *** NOTES ***
            # Other nodes, such as those corresponding to image width and height,
            # might have an increment other than 1. In these cases, it can be
            # important to check that the desired value is a multiple of the
            # increment. However, as these values are being set to the maximum,
            # there is no reason to check against the increment.
            node_width = PySpin.CIntegerPtr(nodemap.GetNode('Width'))
            if PySpin.IsAvailable(node_width) and PySpin.IsWritable(node_width):

                width_to_set = node_width.GetMax()
                node_width.SetValue(width_to_set)
                print('Width set to %i...' % node_width.GetValue())

            else:
                print('Width not available...')

            # Set maximum height
            #
            # *** NOTES ***
            # A maximum is retrieved with the method GetMax(). A node's minimum and
            # maximum should always be a multiple of its increment.
            node_height = PySpin.CIntegerPtr(nodemap.GetNode('Height'))
            if PySpin.IsAvailable(node_height) and PySpin.IsWritable(node_height):

                height_to_set = node_height.GetMax()
                node_height.SetValue(height_to_set)
                print('Height set to %i...' % node_height.GetValue())

            else:
                print('Height not available...')

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False

        return result

