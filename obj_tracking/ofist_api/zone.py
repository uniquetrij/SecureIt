import matplotlib.path as mplPath
import numpy as np

# from utils.csv_reader import read_csv
from utils.csv_reader import read_csv


class Zone:
    def __init__(self, id, coords):
        self.__id = id
        self.__coords = coords
        self.__zone_path = mplPath.Path(np.array(self.__coords))

    def is_in_zone(self, point):
        return self.__zone_path.contains_point(point)

    def is_centroid_in_zone(self, bbox):
        return self.is_in_zone(Zone.find_centroid(bbox))

    def is_bbox_in_zone(self, bbox):
        return self.is_in_zone((bbox[0], bbox[1])) or self.is_in_zone((bbox[2], bbox[1])) or \
               self.is_in_zone((bbox[2], bbox[3])) or self.is_in_zone((bbox[0], bbox[3])) or \
               self.is_centroid_in_zone(bbox)

    def get_id(self):
        return self.__id

    def get_coords(self):
        return self.__coords

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
            coords = line[1:]

            pts =[]
            for i in range(0, len(coords), 2):
                x, y = coords[i], coords[i+1]
                pts.append([float(x),float(y)])
            zones.append(Zone(line[0], np.array(pts)))
        return zones