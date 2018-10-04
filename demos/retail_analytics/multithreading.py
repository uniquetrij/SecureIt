#!/usr/bin/env python
from threading import Thread
import requests
from flask import Flask, render_template, Response
import cv2
import _thread
from data.videos import path as videos_path

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
import json
from data.demos.retail_analytics.inputs import path as input_path
from data.demos.retail_analytics.trained import path as model_path


import matplotlib.pyplot as plt

from demos.retail_analytics.frontend import YOLO
from demos.retail_analytics.utilities import draw_boxes
from tf_session.tf_session_utils import Pipe

app = Flask(__name__)
# app = None

rack_range=None
horizontal_stacks=None
vertical_stacks=None
shelfs_matrix = None
left_x=None
right_x=None
top_y=None
bottom_y=None
shelf_vsize=None
shelf_hsize=None
detergent_range=None
mineralWater_range=None
biscuit_range=None
lays_range=None
noodles_range=None
coke_range=None
product_range={}
shelf_dict={}
labels_dict=None
labels=[]
shelf_state={}
yolo = None
config = None
shelf_product_type=None


image_in_pipe = Pipe()
zone_detection_in_pipe = Pipe()



def global_init(h_stack=3,vstack=2):
    global yolo, config,shelf_product_type
    config_path = input_path.get()+"/config.json"
    weights_path = model_path.get()+"/full_yolo_detergent_and_maggie.h5"
    global detergent_range,mineralWater_range,mineralWater_range,biscuit_range,lays_range,noodles_range,coke_range
    global product_range
    global shelf_hsize,horizontal_stacks
    global vertical_stacks,shelfs_matrix,left_x,right_x,bottom_y,shelf_vsize,top_y,rack_range,labels_dict,labels
    #[(100,20),(1500,20),(100,1050),(1500,1050)]
    rack_range=[(400,205),(1150,205),(400,700),(1150,700)]
    # rack_range=[[428,202],[1143,212],[1116,682],[469,682]]
    horizontal_stacks=h_stack
    vertical_stacks=vstack
    shelfs_matrix = [[None for x in range(vertical_stacks)] for y in range(horizontal_stacks)]
    left_x=rack_range[0][0]
    right_x=rack_range[1][0]
    top_y=rack_range[0][1]
    bottom_y=rack_range[2][1]
    shelf_count=1
    shelf_vsize=(bottom_y-top_y)/horizontal_stacks
    shelf_hsize=(right_x-left_x)//vertical_stacks
    for i in range(0,horizontal_stacks):
        for j in range(0,vertical_stacks):
            shelfs_matrix[i][j]=(j*shelf_hsize+left_x,i*shelf_vsize+top_y)
            shelf_dict["shelf"+str(shelf_count)]=(j*shelf_hsize+left_x,i*shelf_vsize+top_y)
            shelf_count+=1
    labels=[1,2,3,4,5,6]
    detergent_range=shelf_dict["shelf1"]
    mineralWater_range=shelf_dict["shelf2"]
    biscuit_range=shelf_dict["shelf3"]
    lays_range=shelf_dict["shelf4"]
    noodles_range=shelf_dict["shelf5"]
    coke_range=shelf_dict["shelf6"]
    labels_dict={1:"detergent",4:"noodles",0:"lays",2:"mineral_water",3:"coke",5:"biscuit"}
    product_range={2:mineralWater_range,1:detergent_range,5:biscuit_range,4:noodles_range,3:coke_range,0:lays_range}
    shelf_product_type=['detergent','mineral_water','biscuit','lays','noodles','coke']
    #model Load
    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])
    print(weights_path)
    yolo.load_weights(weights_path)
    print("successfull")   

def get_ycordinates(box,image_h, image_w):
    return int(box.ymin*image_h),int(box.ymax*image_h)
def get_xcordinates(box,image_h, image_w):
    return int(box.xmin*image_w),int(box.xmax*image_w)
def misplacedBoxes(boxes,image):
    global shelf_dict
    # print("yes")
    image_h, image_w, _ = image.shape
    misplaced=[]

    for shelf_no,shelf_range  in shelf_dict.items():
        for box in boxes:
            ymin,ymax=get_ycordinates(box,image_h, image_w)
            xmin,xmax=get_xcordinates(box,image_h, image_w)
            centery=(ymin+ymax)/2-5
            centerx=(xmin+xmax)/2
            label=box.get_label()
        # print(labels)
            if(label in labels):
                if not ((product_range[label][1]<centery<product_range[label][1]+shelf_vsize) and
                        (product_range[label][0]<centerx<product_range[label][0]+shelf_hsize  ) ):
                    if(box not in misplaced):
                        misplaced.append(box)
                    

                if((shelf_range[1]<centery<shelf_range[1]+shelf_vsize) and
                        (shelf_range[0]<centerx<shelf_range[0]+shelf_hsize)):
                        # shelf_state[shelf_no]['products'].append(labels_dict[label])
                        shelf_state[shelf_no]['products'][labels_dict[box.get_label()]]+=1

                        if not ((product_range[label][1]<centery<product_range[label][1]+shelf_vsize) and
                        (product_range[label][0]<centerx<product_range[label][0]+shelf_hsize  ) ):
                            # shelf_state[shelf_no]['misplaced'].append(labels_dict[label])
                            shelf_state[shelf_no]['misplaced'][labels_dict[box.get_label()]]+=1

                
                    
                # if(shelf_no=='shelf3'):
                #     cv2.putText(image, 
                #     str(str(shelf_state[shelf_no]['misplaced'])), 
                #         (400,120 ), 
                #         cv2.FONT_HERSHEY_SIMPLEX, 
                #         1e-3 * image_h, 
                #         (0,0,255), 3)
                #     cv2.putText(image, 
                #     str(shelf_state[shelf_no]['products']), 
                #         (400,170 ), 
                #         cv2.FONT_HERSHEY_SIMPLEX, 
                #         1e-3 * image_h, 
                #         (0,255,0), 3)


 
    
    image=draw_box_misplaced(image,misplaced)
    
    return image

def draw_box_misplaced(image,misplaced):
    misplaced_str="misplaced items:"
    image_h, image_w, _ = image.shape
    for product in misplaced:
        misplaced_str+=labels_dict[product.get_label()]+","
        
        ymin,ymax=get_ycordinates(product,image_h, image_w)
        xmin,xmax=get_xcordinates(product,image_h, image_w)
        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,0,255), 3)
        # print("rectangle")
        cv2.putText(image, 
               misplaced_str, 
                (60,60 ), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1e-3 * image_h*1.5, 
                (0,0,255), 3)
    return image
def draw_empty_space(boxes,image):
    global shelf_state
    box_in_shelf=[]
    totalempty_percentage=[]
    image_h, image_w, _ = image.shape
    for shelf_no,shelf_range  in shelf_dict.items():
        empty_space=0
        box_xmin_shelf=[]
        box_xmax_shelf=[]
        for box in boxes:
            ymin,ymax=get_ycordinates(box,image_h, image_w)
            xmin,xmax=get_xcordinates(box,image_h, image_w)
            centery=(ymin+ymax)/2-5
            centerx=(xmin+xmax)/2
            if((shelf_range[1]<centery<shelf_range[1]+shelf_vsize) and
                  (shelf_range[0]<centerx<shelf_range[0]+shelf_hsize)):
                box_xmin_shelf.append(xmin)
                box_xmax_shelf.append(xmax)

                
        
        box_xmin_shelf.append(shelf_range[0]+shelf_hsize)
        box_xmax_shelf.append(shelf_range[0]+shelf_hsize)
        y_box=shelf_range[1]+20
        box_xmin_shelf.sort()
        box_xmax_shelf.sort()
        x_start=shelf_range[0]+20
        x_end=shelf_range[0]+shelf_hsize

        #draw boxes
        for i in range(0,len(box_xmin_shelf)):
            xmin=box_xmin_shelf[i]
            xmax=box_xmax_shelf[i]
            if(xmin-x_start>80):
                cv2.rectangle(image, (int(x_start+5),int(y_box)), 
                              (int(xmin-5),int(y_box+shelf_vsize-10)), (255,0,0), 3)
                empty_space+=xmin-x_start

            x_start=xmax
        empty_percentage=empty_space/shelf_hsize
        # totalempty_percentage.append(empty_percentage)
        # shelf_state[shelf_no]={'perempty':{},'misplaced':{}}
        shelf_state[shelf_no]['perempty']=empty_percentage
  
    return image

def print_shelfNo(image):
    global yolo, config,shelf_product_type
    global shelf_state
    shelf_count=1
    image_h, image_w, _ = image.shape
    for i in range(0,horizontal_stacks):
        for j in range(0,vertical_stacks):
            cv2.putText(image, 
                str(shelf_count), 
                    (int(j*shelf_hsize+left_x+55),int(i*shelf_vsize+top_y+75) ), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1e-3 * image_h*1.5, 
                    (0,0,255), 3)
            shelf_state['shelf'+str(shelf_count)]={'perempty':None,'misplaced':{'lays':0,'detergent':0,'coke':0,'mineral_water':0,'biscuit':0,'noodles':0},
            'products':{'lays':0,'detergent':0,'coke':0,'mineral_water':0,'biscuit':0,'noodles':0},
            'position':shelf_count-1,'product_type':shelf_product_type[shelf_count-1]}
            shelf_count+=1
    return image

# def post_dict_process():
#     global shelf_state
#     # for shelf_no,shelf_desc in shelf_state.items():
#     #     for product in shelf_desc['misplaced']:
#     #         shelf_state[shelf_no]['misplaced'][labels_dict[product.get_label()]]+=1
#     # for shelf_no,shelf_desc in shelf_state.items():
#     #     for product in shelf_desc['products']:
#     #         shelf_state[shelf_no]['products'][labels_dict[product.get_label()]]+=1
#     # print(shelf_state)
    

    
    







def postdata():
    global shelf_state
    #res = requests.post('http://localhost:5000/tests/endpoint', json=shelf_state)
    print('response from server:',shelf_state)

# @app.route('/')
# def index():
#     return render_template('index.html')

def gen():
    global yolo
    global_init()
    # cap = cv2.VideoCapture(videos_path.get() + '/ra_rafee_cabin_1.mp4')
    while True:
        # Capture frame-by-frame
        # ret, image = cap.read()
        image_in_pipe.wait()
        ret, image = image_in_pipe.pull()
        if not ret:
            continue




        M = cv2.getPerspectiveTransform(np.array([[428,202],[1143,212],[1116,682],[469,682]],dtype="float32"),
        np.array([[428,202],[1143,202],[1143,682],[428,682]],dtype="float32"))
        image = cv2.warpPerspective(image, M, (image.shape[1],image.shape[0]))
        boxes = yolo.predict(image)
        image = draw_boxes(image, boxes, config['model']['labels'])
        ret, zones = zone_detection_in_pipe.pull()
        if ret and not zones:
            image=print_shelfNo(image)
            image=misplacedBoxes(boxes,image)
            image=draw_empty_space(boxes,image)
            # postdatathread = Thread(target=postdata,args = ())
            # postdatathread.start()
            # _thread.start_new_thread( tick, ( ) )
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', image)[1].tostring() + b'\r\n')

@app.route('/')
def video_feed():
    print("Video Feed")
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def run():
    app.run(host='0.0.0.0', debug=True, use_reloader=False)

if __name__ == '__main__':
    # scheduler = BackgroundScheduler()
    # scheduler.add_job(tick, 'interval', seconds=3)
    # scheduler.start()
    app.run(host='0.0.0.0',debug=True, use_reloader=False)
    # try:
    #     # This is here to simulate application activity (which keeps the main thread alive).
    #     while True:
    #         time.sleep(2)
    # except (KeyboardInterrupt, SystemExit):
    #     # Not strictly necessary if daemonic mode is enabled but should be done if possible
    #     scheduler.shutdown()
