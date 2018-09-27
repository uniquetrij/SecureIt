import matplotlib.path as mplPath
import numpy as np

from utils.csv_reader import read_csv


class Zone:
    def __init__(self, co_ordinates):
        self.__coords = co_ordinates
        self.__zone_path = mplPath.Path(np.array(self.__coords))

    def is_in_zone(self, point):
        return self.__zone_path.contains_point(point)

    def is_centroid_in_zone(self, bbox):
        return self.is_in_zone(Zone.find_centroid(bbox))

    @staticmethod
    def find_centroid(bbox):
        cX = int((bbox[0] + bbox[2]) / 2)
        cY = int((bbox[1] + bbox[3]) / 2)
        return (cX, cY)

    @staticmethod
    def create_zones_from_conf(conf_path):
        zone_conf = read_csv(conf_path)
        zones = []
        for line in zone_conf:
            cols = line
            print(line)
            pts =[]
            for i in range(0, len(cols), 2):
                x, y = cols[i], cols[i+1]
                pts.append([float(x),float(y)])
            zones.append(Zone(np.array(pts)))
        return zones