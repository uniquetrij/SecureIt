import time

from demos.retail_analytics.zone import Zone
from data.demos.retail_analytics.inputs import path as file_path


class Trail:
    conf_path = file_path.get() + '/zones.csv'
    zones = Zone.create_zones_from_conf(conf_path)
    def __init__(self, id):
        self.__tracker_id = id
        self.__in_time = None
        self.__curr_zone = None
        self.__prev_zone = None
        self.__regions = []
        self.__track = []
        self.__entries = {}
        self.__exits = {}
        self.__exit_threshold = 3
        # self.__bboxes = []
    # def __str__:
    #     return
    # def update_track(self, bbox):
    #     self.__curr_zone = None
    #     for i, zone in enumerate(Trail.zones):
    #         if zone.is_centroid_in_zone(bbox):
    #             self.__curr_zone = i
    #             # print("In zone", self.__entred_zone)
    #             self.__bboxes.append(bbox)
    #             if self.__prev_zone is None:
    #                 print(self.__tracker_id, "Entered Zone.")
    #                 self.__in_time = time.time()
    #                 self.__prev_zone = self.__curr_zone
    #             else:
    #                 if self.__curr_zone is not self.__prev_zone:
    #                     print("Exited Zone and Entered a new Zone")
    #                     self.__track.append((i, self.__in_time, time.time(), self.__bboxes))
    #                     print(self.__tracker_id, self.__track)
    #                     self.__bboxes = []
    #                     self.__in_time = None
    #
    #     if self.__curr_zone is None and self.__prev_zone is not None:
    #         print(self.__tracker_id, "Exited Zone")
    #         self.__track.append((i, self.__in_time, time.time(), self.__bboxes))
    #         # print(self.__tracker_id, self.__track)
    #         self.__bboxes = []
    #         self.__in_time = None
    #         self.__prev_zone = None

    def update_track(self, bbox):
        self.__curr_zone = None
        timestamp = time.time()

        for i, zone in enumerate(Trail.zones):
            if zone.is_centroid_in_zone(bbox):
                self.__curr_zone = i
                if self.__curr_zone not in self.__entries:
                    print(self.__tracker_id,"Entered Zone")
                    self.__entries[self.__curr_zone] = [(len(self.__track),timestamp)]
                    self.__exits[self.__curr_zone] = [(len(self.__track),timestamp)]
                else:
                    prev = self.__exits[self.__curr_zone][-1][1]
                    if timestamp - prev > self.__exit_threshold:
                        print(self.__tracker_id,"Entered Zone")
                        self.__entries[self.__curr_zone].append((len(self.__track),timestamp))
                        self.__exits[self.__curr_zone].append((len(self.__track), timestamp))
                    else:
                        self.__exits[self.__curr_zone][-1] = (len(self.__track),timestamp)
                        print(self.__tracker_id,"Exiting....")
                break

        self.__track.append((self.__curr_zone, timestamp, bbox))

    def get_zone(self):
        return self.__curr_zone

    def get_trail(self):
        return self.__track

