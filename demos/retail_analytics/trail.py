import time


class Trail:

    def __init__(self, zones, id):
        self.__zones = zones
        self.__tracker_id = id
        self.__in_time = None
        self.__prev_zone = None
        self.__regions = []
        self.__track = []
        self.__entries = {}
        self.__exits = {}
        self.__exit_threshold = 1

    def update_track(self, bbox):
        self.__curr_zones = []
        timestamp = time.time()

        for zone in self.__zones:
            if zone.is_centroid_in_zone(bbox):
                index = zone.get_id()
                self.__curr_zones.append(zone)
                if index not in self.__entries:
                    self.__entries[index] = [len(self.__track)]
                    self.__exits[index] = [len(self.__track)]
                else:
                    prev = self.__track[self.__exits[index][-1]][0]
                    if timestamp - prev > self.__exit_threshold:
                        self.__entries[index].append(len(self.__track))
                        self.__exits[index].append(len(self.__track))
                    else:
                        self.__exits[index][-1] = len(self.__track)

        self.__track.append((timestamp, bbox))

    def get_current_zones(self):
        return self.__curr_zones

    def get_trail(self):
        return self.__track
