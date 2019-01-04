import time
import requests
from person.person_metadata import Person

API_ENDPOINT = "http://192.168.43.55:5000/notify_zone_entry"

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
        # print(type(person))
        self.__person = Person()

    def get_person(self):
        return self.__person

    def update_zones(self, zones):
        self.__zones = zones
        self.__prev_zones = []
        self.__regions = []
        self.__track = []
        self.__entries = {}
        self.__exits = {}

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
                    print("Zone: "+str(index)+" by person "+str(self.__tracker_id))
                    response = {'{},{}'.format(index, str(self.__tracker_id))}

                    self.post_response(str(index), str(self.__tracker_id))

                    self.__entries[index] = [len(self.__track)]
                    self.__exits[index] = [len(self.__track)]
                else:
                    prev = self.__track[self.__exits[index][-1]][0]
                    if timestamp - prev > self.__exit_threshold:
                        response = {'{},{}'.format(zone.get_id(),str(self.__tracker_id) )}
                        # response_pickled = jsonpickle.encode(response)
                        print("Re-entry in zone "+zone.get_id()+" by person "+str(self.__tracker_id))

                        self.post_response(zone.get_id(), str(self.__tracker_id))

                        self.__entries[index].append(len(self.__track))
                        self.__exits[index].append(len(self.__track))
                    else:
                        # print("exited")
                        self.__exits[index][-1] = len(self.__track)

    def post_response(self,zone,person_id):
        person_gender = None
        person_age = None
        person_age_list = self.__person.get_age_list()
        person_gender_list = self.__person.get_age_list()
        gender_on_confidence_score = self.__person.get_gender_on_confidence_score()
        if len(person_age_list) != 0:
            person_age = int(sum(person_age_list) / len(person_age_list))
        if len(person_gender_list) != 0:
            if person_gender_list.count('M') == person_gender_list.count('F'):
                person_gender = gender_on_confidence_score
            else:
                person_gender = max(person_gender_list, key=person_age_list.count)

        data = {
            'zone_id': zone,
            'person_id':person_id,
            'person_age': str(person_age) if person_age else "Unknown",
            'person_gender': person_gender if person_gender else "Unknown"
        }
        # r=requests.post(url=API_ENDPOINT, json=data)
        # print("Entry detected :"+str(r))
        print(data)

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
