import cv2
import requests
from data.demos.retail_analytics.inputs import path as input_path
import json

class RetailAnalytics():
    _check = 0
    def __init__(self):
        self.rack_range = None
        self.horizontal_stacks = None
        self.vertical_stacks = None
        self.shelfs_matrix = None
        self.shelf_vsize = None
        self.shelf_hsize = None
        self.shelf_dict = {}
        self.labels_dict = None
        self.labels = []
        self.shelf_state = {}
        self.yolo = None
        self.config = None
        self.shelf_product_type = None
        self.prev_shelf_state_1 = None
        self.postjsondata = {}
        self.finaljsondata = {}
        self.check_flag = 0
        self.rack_dict = None
        self.product_shelf = None
        self.left_x = None
        self.right_x = None
        self.top_y = None
        self.bottom_y = None
        self.promo_shelf = None
        self.promo_number = None
        self.promo_check = False

    def global_init(self, session_runner=None):
        self.config_path = input_path.get() + "/config.json"
        with open(self.config_path) as config_buffer:
            self.config = json.load(config_buffer)
        rack_cord = None
        check_flag = 0
        if (self.rack_dict != None):
            if (self.rack_dict['point_set_2'] != None):
                rack_cord = self.rack_dict['point_set_2'].copy()
                check_flag = 1
                temp = rack_cord[2]
                rack_cord[2] = rack_cord[3]
                rack_cord[3] = temp

        if (check_flag == 1):
            self.horizontal_stacks = self.config["global_init"]["h_stack"]
            self.vertical_stacks = self.config["global_init"]["v_stack"]
            self.promo_number = self.config["global_init"]["promo_shelf"]
            self.promo_check = self.config["global_init"]["is_promo"]
            self.shelfs_matrix = [[None for x in range(self.vertical_stacks)] for y in range(self.horizontal_stacks)]
            self.left_x = rack_cord[0][0]
            self.right_x = rack_cord[1][0]
            self.top_y = rack_cord[0][1]
            self.bottom_y = rack_cord[2][1]
            shelf_count = 1
            self.shelf_vsize = (self.bottom_y - self.top_y) / self.horizontal_stacks
            self.shelf_hsize = (self.right_x - self.left_x) // self.vertical_stacks
            offset = 0
            for i in range(0, self.horizontal_stacks):
                for j in range(0, self.vertical_stacks):
                    self.shelfs_matrix[i][j] = (
                        j * self.shelf_hsize + self.left_x, i * self.shelf_vsize + self.top_y + offset)
                    self.shelf_dict["shelf" + str(shelf_count)] = (
                        j * self.shelf_hsize + self.left_x, i * self.shelf_vsize + self.top_y + offset)
                    shelf_count += 1

            self.shelf_product_type = ['detergent', 'mineral_water', 'biscuit', 'lays', 'noodles', 'coke']
            self.labels_dict = {1: "detergent", 4: "noodles", 0: "lays", 2: "mineral_water", 3: "coke", 5: "biscuit"}

            # shelfno: product label
            self.product_shelf = {1: 1, 2: 2, 3: 5, 4: 0, 5: 4, 6: 3}
            print("successfull")
            return True
        else:
            return False

    def get_ycordinates(self,box,image_h, image_w):
        xmin, ymin, xmax, ymax= box.get_bbox()
        return int(ymin*image_h),int(ymax*image_h)

    def get_xcordinates(self,box,image_h, image_w):
        xmin, ymin, xmax, ymax = box.get_bbox()
        return int(xmin*image_w),int(xmax*image_w)

    def misplacedBoxes(self, boxes, image):
        image_h, image_w, _ = image.shape
        misplaced = []

        for shelf_no, shelf_range in self.shelf_dict.items():
            for box in boxes:
                ymin, ymax = self.get_ycordinates(box, image_h, image_w)
                xmin, xmax = self.get_xcordinates(box, image_h, image_w)
                centery = (ymin + ymax) / 2 - 5
                centerx = (xmin + xmax) / 2
                label = box.get_label()
                num = int(shelf_no[5:])
                if (self.product_shelf[num] != box.get_label() and (
                        (shelf_range[1] < centery < shelf_range[1] + self.shelf_vsize) and
                        (shelf_range[0] < centerx < shelf_range[0] + self.shelf_hsize))):
                    if (box not in misplaced):
                        misplaced.append(box)
                        if self.labels_dict[box.get_label()] in self.shelf_state[shelf_no]['misplaced']:
                            self.shelf_state[shelf_no]['misplaced'][self.labels_dict[box.get_label()]] += 1
                        else:
                            self.shelf_state[shelf_no]['misplaced'][self.labels_dict[box.get_label()]] = 1

                else:
                    if (((shelf_range[1] < centery < shelf_range[1] + self.shelf_vsize) and
                         (shelf_range[0] < centerx < shelf_range[0] + self.shelf_hsize)) and self.product_shelf[
                        num] == box.get_label()):
                        if self.labels_dict[box.get_label()] in self.shelf_state[shelf_no]['products']:
                            self.shelf_state[shelf_no]['products'][self.labels_dict[box.get_label()]] += 1
                        else:
                            self.shelf_state[shelf_no]['products'][self.labels_dict[box.get_label()]] = 1

        image = self.draw_box_misplaced(image, misplaced)

        return image

    def draw_box_misplaced(self,image,misplaced):
        misplaced_str="misplaced items:"
        image_h, image_w, _ = image.shape
        for product in misplaced:
            misplaced_str+=self.labels_dict[product.get_label()]+","

            ymin,ymax=self.get_ycordinates(product,image_h, image_w)
            xmin,xmax=self.get_xcordinates(product,image_h, image_w)
            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,0,255), 3)
            # cv2.putText(image,
            #        misplaced_str,
            #         (30,30 ),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         1e-3 * image_h*1.5,
            #         (0,0,255), 3)
        return image
    def draw_empty_space(self,boxes,image):
        image_h, image_w, _ = image.shape
        for shelf_no,shelf_range  in self.shelf_dict.items():
            empty_space=0
            box_xmin_shelf=[]
            box_xmax_shelf=[]
            for box in boxes:
                ymin,ymax=self.get_ycordinates(box,image_h, image_w)
                xmin,xmax=self.get_xcordinates(box,image_h, image_w)
                centery=(ymin+ymax)/2-5
                centerx=(xmin+xmax)/2
                if((shelf_range[1]<centery<shelf_range[1]+self.shelf_vsize) and
                      (shelf_range[0]<centerx<shelf_range[0]+self.shelf_hsize)):
                    box_xmin_shelf.append(xmin)
                    box_xmax_shelf.append(xmax)



            box_xmin_shelf.append(shelf_range[0]+self.shelf_hsize)
            box_xmax_shelf.append(shelf_range[0]+self.shelf_hsize)
            y_box=shelf_range[1]+5
            box_xmin_shelf.sort()
            box_xmax_shelf.sort()
            x_start=shelf_range[0]
            x_end=shelf_range[0]+self.shelf_hsize

            #draw boxes
            for i in range(0,len(box_xmin_shelf)):
                xmin=box_xmin_shelf[i]
                xmax=box_xmax_shelf[i]
                if(xmin-x_start>80):
                    cv2.rectangle(image, (int(x_start+5),int(y_box)),
                                  (int(xmin-5),int(y_box+self.shelf_vsize)), (255,0,0), 3)
                    empty_space+=xmin-x_start
                else:
                    empty_space+=xmin-x_start
                x_start=xmax
            empty_percentage=empty_space/self.shelf_hsize
            self.shelf_state[shelf_no]['perempty']=empty_percentage

        return image

    def print_shelfNo(self, image):
        shelf_count = 1
        image_h, image_w, _ = image.shape
        for i in range(0, self.horizontal_stacks):
            for j in range(0, self.vertical_stacks):
                cv2.putText(image, str(shelf_count),
                            (int(j * self.shelf_hsize + self.left_x + 55), int(i * self.shelf_vsize + self.top_y + 45)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1e-3 * image_h * 1,
                            (0, 0, 255), 2)
                self.shelf_state['shelf' + str(shelf_count)] = {'perempty': None, 'misplaced': {},
                                                                'products': {},
                                                                'position': shelf_count - 1,
                                                                'product_type': self.shelf_product_type[
                                                                    shelf_count - 1]}
                shelf_count += 1
        # shelf_count = 1
        # for i in range(0, len(self.product_shelf)):
        #     cv2.putText(image, str(shelf_count) + ":" + self.labels_dict[self.product_shelf[shelf_count]],
        #                 ((image_w - 500), shelf_count * 50 + 100),
        #                 cv2.FONT_HERSHEY_SIMPLEX,
        #                 1e-3 * image_h * 1,
        #                 (0, 0, 255), 2)
        #     shelf_count += 1

        # self.labels_dict[self.product_shelf[shelf_count]]
        return image

    def change_of_state(self):
        flag = 0
        promo_flag = 0
        if (self.prev_shelf_state_1 == None):
            self.prev_shelf_state_1 = self.shelf_state
            flag = 1
            if (self.promo_check):
                self.promo_shelf = self.shelf_state['shelf' + str(self.promo_number)].copy()
                # print(self.promo_shelf)
                if 'products' in self.promo_shelf:
                    del self.promo_shelf['products']
                    promo_flag = 1

        for i in range(1, len(self.shelf_dict) + 1):
            per = self.shelf_state['shelf' + str(i)]['perempty']
            pre_per = self.prev_shelf_state_1['shelf' + str(i)]['perempty']
            if (abs(pre_per - per) > 20):
                print("empty")
                flag = 1
                break

            current_list = self.shelf_state['shelf' + str(i)]['misplaced'].keys()
            previous_list = self.prev_shelf_state_1['shelf' + str(i)]['misplaced'].keys()

            count = 0
            if (len(previous_list) == len(current_list)):
                for x in previous_list:
                    if x in current_list:
                        count += 1
            else:
                flag = 1
                break

            if (count != len(current_list)):
                flag = 1
                break
            else:
                for ch_products in self.shelf_state['shelf' + str(i)]['misplaced']:
                    if((self.shelf_state['shelf' + str(i)]['misplaced'][ch_products] - self.prev_shelf_state_1['shelf' + str(i)]['misplaced'][ch_products]) != 0):
                        flag = 1
                        break

        if (flag == 1 and self.promo_check):
            promo_previous_list = self.promo_shelf['misplaced'].keys()
            promo_current_lits = self.shelf_state['shelf' + str(self.promo_number)]['misplaced'].keys()
            # print(promo_previous_list)
            # print(promo_current_lits)
            counter = 0
            if (len(promo_previous_list) == len(promo_current_lits)):
                for x in promo_previous_list:
                    if x in promo_current_lits:
                        counter += 1
            else:
                promo_flag = 1

            if(promo_flag != 1):
                if (counter != len(promo_current_lits)):
                    promo_flag = 1
                else:
                    for ch_products in self.promo_shelf['misplaced']:
                        if((self.shelf_state['shelf' + str(self.promo_number)]['misplaced'][ch_products] - self.promo_shelf['misplaced'][ch_products]) != 0):
                            promo_flag = 1
                            break
            # print(promo_flag,self.promo_check)

        if (flag == 1):
            tempJson = []
            for i in self.shelf_state.keys():
                tempJson.append({"name": i, "value": self.shelf_state[i]})
                # t5t13 #8glq
            self.postjsondata = {'store':"store-2jmvt5t13",'rack':"rack-2jmvtheoo",'zone':"zone-2jmvts25j",'shelves':tempJson}
            # print(res)
            # s = json.dumps(self.postjsondata)
            print(self.postjsondata, ' ', RetailAnalytics._check, '\n')
            # open("out.json", "w").write(s+'\n')
            RetailAnalytics._check += 1

        if (promo_flag == 1 and self.promo_check):
            promoJson = self.shelf_state['shelf' + str(self.promo_number)].copy()
            if 'products' in promoJson:
                del promoJson['products']
            # print('in')

            print(promoJson)
            # self.postjsondata = {'store': "store-2jmvt5t13", 'rack': "rack-2jmvtheoo", 'zone': "zone-2jmvts25j",
            #                      'shelves': self.promo_shelf}

        self.prev_shelf_state_1 = self.shelf_state.copy()
        self.promo_shelf = self.shelf_state['shelf' + str(self.promo_number)].copy()

        if 'products' in self.promo_shelf:
            del self.promo_shelf['products']
        # print(self.promo_shelf, 'test1')
        return flag

    def postdata(self):
        res = requests.post('https://us-central1-retailanalytics-d6ccf.cloudfunctions.net/api/misplaced-items',json=self.postjsondata)