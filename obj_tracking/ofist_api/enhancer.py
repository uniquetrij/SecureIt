import cv2

from obj_tracking.ofist_api import retinex
import numpy as np


class ImageEnhancer:
    retinex_conf = {
        "sigma_list": [15, 80, 250],
        "G": 5.0,
        "b": 25.0,
        "alpha": 125.0,
        "beta": 46.0,
        "low_clip": 0.01,
        "high_clip": 0.99
    }

    @staticmethod
    def preprocess_bright_contrast(image, brightness=50, contrast=30):
        image = np.int16(image.copy())
        image = image * (contrast / 127 + 1) - contrast + brightness
        image = np.clip(image, 0, 255)
        image = np.uint8(image)
        return image

    @staticmethod
    def preprocess_hist_eq(image):
        image = image.copy()
        image[0] = cv2.equalizeHist(image[0])
        image[1] = cv2.equalizeHist(image[1])
        image[2] = cv2.equalizeHist(image[2])
        return image

    @staticmethod
    def preprocess_retinex(image):
        image = retinex.MSRCP(image, ImageEnhancer.retinex_conf['sigma_list'],
                              ImageEnhancer.retinex_conf['low_clip'],
                              ImageEnhancer.retinex_conf['high_clip'])
        # print(np.max(image))
        # image = retinex.singleScaleRetinex(image, 80)
        # print(image.shape)
        return image

    @staticmethod
    def gamma_correction(image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    @staticmethod
    def hls_enhancement(image, h=1, l=1, s=1):
        cvt_img = cv2.cvtColor(image, cv2.COLOR_BGR2HLS).astype("float32")
        (_h, _l, _s) = cv2.split(cvt_img)
        _h *= h
        _s *= s
        _l *= l
        _h = np.clip(_h, 0, 255)
        _l = np.clip(_l, 0, 255)
        _s = np.clip(_s, 0, 255)
        cvt_img = cv2.merge([_h, _l, _s])
        out_img = cv2.cvtColor(cvt_img.astype("uint8"), cv2.COLOR_HLS2BGR)
        return out_img

    @staticmethod
    def hsv_enhancement(image, h=1, s=1, v=1):
        cvt_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype("float32")
        (_h, _s, _v) = cv2.split(cvt_img)
        _h *= h
        _s *= s
        _v *= v
        _h = np.clip(_h, 0, 255)
        _s = np.clip(_s, 0, 255)
        _v = np.clip(_v, 0, 255)
        cvt_img = cv2.merge([_h, _s, _v])
        out_img = cv2.cvtColor(cvt_img.astype("uint8"), cv2.COLOR_HSV2BGR)
        return out_img

    @staticmethod
    def lab_enhancement(image, l=1, a=1, b=1):
        cvt_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype("float32")
        (_l, _a, _b) = cv2.split(cvt_img)
        _l *= l
        _a *= a
        _b *= b
        _l = np.clip(_l, 0, 255)
        _a = np.clip(_a, 0, 255)
        _b = np.clip(_b, 0, 255)
        cvt_img = cv2.merge([_l, _a, _b])
        out_img = cv2.cvtColor(cvt_img.astype("uint8"), cv2.COLOR_LAB2BGR)
        return out_img


    @staticmethod
    def gaussian_blurr(image, sigma):
        return cv2.GaussianBlur(image, (0, 0), sigma)

    @staticmethod
    def selective_enhancement(image, h=0, s=0, v=0):
        cvt_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype("float32")
        (_h, _s, _v) = cv2.split(cvt_img)

        _v = np.add(_v, np.divide(np.subtract(np.float32(np.multiply(255, np.ones(_v.shape))), _v), 4))

        # for row in _v:
        #     for i in range(len(row)):
        #         row[i] += (255 - row[i])/4
        # for row in _s:
        #     for i in range(len(row)):
        #         row[i] += (255 - row[i])/4

        _v = np.clip(_v, 0, 255)
        # _s = np.clip(_s, 0, 255)
        cvt_img = cv2.merge([_h, _s, _v])
        out_img = np.uint8(cv2.cvtColor(cvt_img.astype("uint8"), cv2.COLOR_HSV2BGR))
        print(out_img.shape)
        return out_img


