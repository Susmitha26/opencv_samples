            

import cv2
import os
import sys
import math

################################################################################

import tflearn
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.normalization import *
from tflearn.layers.estimator import regression

################################################################################

def construct_firenet (x,y, training=False):

    # Build network as per architecture in [Dunnings/Breckon, 2018]

    network = tflearn.input_data(shape=[None, y, x, 3], dtype=tf.float32)

    network = conv_2d(network, 64, 5, strides=4, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 128, 4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 256, 1, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = fully_connected(network, 4096, activation='tanh')
    if(training):
        network = dropout(network, 0.5)

    network = fully_connected(network, 4096, activation='tanh')
    if(training):
        network = dropout(network, 0.5)

    network = fully_connected(network, 2, activation='softmax')

    # if training then add training hyperparameters

    if(training):
        network = regression(network, optimizer='momentum',
                            loss='categorical_crossentropy',
                            learning_rate=0.001)

    # constuct final model

    model = tflearn.DNN(network, checkpoint_path='firenet',
                         tensorboard_verbose=2)

    return model

################################################################################
#def detect_fire(frame,frame_no,time):
if __name__ == '__main__':
    # construct and display model
   model = construct_firenet (224, 224, training=False)
   # print("Constructed FireNet ...")

   model.load(os.path.join("models/FireNet", "firenet"),weights_only=True)
    #print("Loaded CNN network weights ...")

################################################################################

    # network input sizes

   rows = 224
   cols = 224
   windowName = "Live Fire Detection - FireNet CNN"
   keepProcessing = True
   final={}

################################################################################
   video = cv2.VideoCapture('crackersfire1.mp4')
   print("Loaded video ...")

        # create window

   cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

        # get video properties
   width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
   height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
   fps = video.get(cv2.CAP_PROP_FPS)
   frame_time = round(1000/fps)

   while (keepProcessing):

            # start a timer (to see how long processing and display takes)

            start_t = cv2.getTickCount()

            # get video frame from file, handle end of file

            ret, frame = video.read()
            if not ret:
                print("... end of video file reached")
                break


            # re-size image to network input size and perform prediction
            small_frame = cv2.resize(frame, (rows, cols), cv2.INTER_AREA)
            output = model.predict([small_frame])
   #print(output)
            # label image based on prediction
            if round(output[0][0]) == 1:
                    cv2.rectangle(frame, (0,0), (width,height), (0,0,255), 50)
                    cv2.putText(frame,'FIRE',(int(width/16),int(height/4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA)
                    frame_count=int(video.get(cv2.CAP_PROP_POS_FRAMES))
                    time=video.get(cv2.CAP_PROP_POS_MSEC)/1000
	   
                    final['frame']=frame_count
                    final['fire']=1
                    final['time']=time
            else:
                cv2.rectangle(frame, (0,0), (width,height), (0,255,0), 50)
                cv2.putText(frame,'CLEAR',(int(width/16),int(height/4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA)
                frame_count=int(video.get(cv2.CAP_PROP_POS_FRAMES))
                time=video.get(cv2.CAP_PROP_POS_MSEC)/1000
	   
                final['frame']=frame_count
                final['fire']=0
                final['time']=time
            print(final)

