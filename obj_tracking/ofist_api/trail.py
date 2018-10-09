import time


class Trail:

    def __init__(self, zones = None, id = None):
        self.__zones = zones
        self.__tracker_id = id
        self.__in_time = None
        self.__prev_zones = []
        self.__regions = []
        self.__track = []
        self.__entries = {}
        self.__exits = {}
        self.__exit_threshold = 1

    def update_track(self, bbox):
        timestamp = time.time()
        self.__track.append((timestamp, bbox))
        if self.__zones is not None:
            self.__update_zone(bbox, timestamp)

    def __update_zone(self, bbox, timestamp):
        self.__curr_zones = []
        for zone in self.__zones:
            if zone.is_bbox_in_zone(bbox):
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

    def get_current_zones(self):
        return self.__curr_zones

    def get_trail(self):
        return self.__track

    def get_exited_zones(self):
        exited_zones = []
        for zone in self.__prev_zones:
            if zone not in self.__curr_zones:
                exited_zones.append(zone)

        self.__prev_zones = self.__curr_zones
        return exited_zones

    def get_exit(self, zone_id):
        return self.__track[self.__exits[zone_id][-1]][0]

    def get_entry(self, zone_id):
        return self.__track[self.__entries[zone_id][-1]][0]
