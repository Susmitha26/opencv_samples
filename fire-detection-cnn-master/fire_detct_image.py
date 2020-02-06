
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
                        max_checkpoints=1, tensorboard_verbose=2)

    return model

################################################################################

if __name__ == '__main__':

################################################################################

    # construct and display model

    model = construct_firenet (224, 224, training=False)
    print("Constructed FireNet ...")

    model.load(os.path.join("models/FireNet", "firenet"),weights_only=True)
    print("Loaded CNN network weights ...")

################################################################################

    # network input sizes

    rows = 224
    cols = 224

    # display and loop settings

    windowName = "Live Fire Detection - FireNet CNN";
    keepProcessing = True;

################################################################################

        # load video file from first command line argument

    img=cv2.imread('fire2.jpg')
        #print("Loaded video ...")

        # create window

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL);

        # get video properties

        #width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH));
        #height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width,height=img.size()
        #fps = video.get(cv2.CAP_PROP_FPS)
        #frame_time = round(1000/fps);

        #while (keepProcessing):

            # start a timer (to see how long processing and display takes)

    start_t = cv2.getTickCount()

            # get video frame from file, handle end of file

           # ret, frame = video.read()
           # if not ret:
              #  print("... end of video file reached");
                #break;

            # re-size image to network input size and perform prediction

    small_frame = img.resize((rows, cols))
    output = model.predict([small_frame])

            # label image based on prediction

    if round(output[0][0]) == 1:
            cv2.rectangle(img, (0,0), (int(width/2),int(height/2)), (0,0,255), 50)
            cv2.putText(img,'FIRE',(int(width/16),int(height/4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA)
    else:
            cv2.rectangle(img, (0,0), (int(width/2),int(height/2)), (0,255,0), 50)
            cv2.putText(img,'CLEAR',(int(width/16),int(height/4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA)

            # stop the timer and convert to ms. (to see how long processing and display takes)

    stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000

            # image display and key handling

    cv2.imshow(windowName, img)
    cv2.waitKey(0)

            # wait fps time or less depending on processing time taken (e.g. 1000ms / 25 fps = 40 ms)

           # key = cv2.waitKey(max(2, frame_time - int(math.ceil(stop_t)))) & 0xFF;
            #if (key == ord('x')):
               # keepProcessing = False;
            #lif (key == ord('f')):
              #  cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
       # print("usage: python firenet.py videofile.ext")