import cv2
import numpy as np
import torch

import imutils

import time
#import matplotlib.pyplot as plt

def capture_webcam(input_cam = 1):
    cv2.namedWindow("Camera ", input_cam)
    vc = cv2.VideoCapture(input_cam)
    ORANGE_MIN = (5, 50, 50)
    ORANGE_MAX = (15, 255, 255)
    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
    centers = []
    radii = []
    
    while rval:
        
        rval, frame = vc.read()

        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, ORANGE_MIN, ORANGE_MAX)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        key = cv2.waitKey(0)
        #cv2.imshow('mask', mask)
        # only proceed if at least one contour was found
        # find contours in the mask and initialize the current
	    # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        #cnts = imutils.grab_contours(cnts)
        center = None

        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # only proceed if the radius meets a minimum size
            if radius > 10:
                centers.append(center)
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
                    (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
        cv2.imshow("Camera 1", frame)
        if key == 27: # exit on ESC
            break

    smoothed_path = plot_xy_ball_path(centers)
    exit_velocity_x, exit_velocity_y = get_xy_exit_velocities(smoothed_path)

    vc.release()
    cv2.destroyWindow("preview")

def plot_xy_ball_path(centers):
    #plt.plot(centers)
    #plt.show()
    pass

def plot_xy_ball_height(radii):
    pass

def get_xy_exit_velocities(smoothed_path):
    pass

def get_launch_angle(radius_function):
    pass

def get_exit_velocity_z(radius_function):
    pass



if __name__ == "__main__":
    capture_webcam()