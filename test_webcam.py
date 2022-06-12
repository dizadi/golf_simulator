import cv2
import numpy as np
import torch

import imutils

import time
import matplotlib.pyplot as plt

def capture_webcam(input_cam = 1):
    cv2.namedWindow("Camera ", input_cam)
    vc = cv2.VideoCapture(input_cam)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
    images = []
    while rval:
        
        rval, frame = vc.read()
        key = cv2.waitKey(1)
        cv2.imshow("Camera 1", frame)
        images.append(frame)

        if key == 27: # exit on ESC
            break
            
    vc.release()
    cv2.destroyWindow("preview")

    initial_location, initial_frame = detect_ball_initial_location(images)
    centers, radii = track_ball(images, initial_location, initial_frame)

def detect_ball_initial_location(images):
    ORANGE_MIN = (5, 150, 150)
    ORANGE_MAX = (15, 255, 255)
    n = 0
    for frame in images:
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, ORANGE_MIN, ORANGE_MAX)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None
        #cv2.imshow('Mask', mask)
        #cv2.waitKey(1)
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
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                initial_location = center
                initial_frame = n
                return initial_location, initial_frame
            n += 1

def track_ball(images, initial_location, initial_frame):
    ORANGE_MIN = (5, 100, 100)
    ORANGE_MAX = (15, 255, 255)
    centers, radii = [], []

    current_location = initial_location
    cv2.namedWindow('Tracking')
    for frame in images[initial_frame:]:
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, ORANGE_MIN, ORANGE_MAX)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
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
            if radius > 5:
                if ((current_location[0]-center[0])**2 + (current_location[1]-center[1])**2)**0.5 <= 10000:
                    centers.append(center)
                    radii.append(radius)
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    cv2.circle(frame, (int(x), int(y)), int(radius),
                        (0, 255, 255), 2)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)
                    cv2.imshow('Tracking', frame)
                    cv2.waitKey(1)
    plot_xy_ball_path(centers[:-10])
    plot_xy_ball_height(radii[:-10])

    return centers, radii
    
def plot_xy_ball_path(centers):
    x = [c[0] for c in centers]
    y = [c[1] for c in centers]
    plt.plot(x, y)
    plt.title('center')
    plt.show()
    

def plot_xy_ball_height(radii):
    plt.plot(radii)
    plt.title('radius')
    plt.show()

def get_xy_exit_velocities(smoothed_path):
    pass

def get_launch_angle(radius_function):
    pass

def get_exit_velocity_z(radius_function):
    pass



if __name__ == "__main__":
    images = []
    test_file='tmp/test.m4v'
    cap = cv2.VideoCapture(test_file)

    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    print('Video opened. Reading images. ')
    print(test_file)
    ret, frame = cap.read()
    #cv2.namedWindow('image')
    while(cap.isOpened()):
        if ret:
            ret, frame = cap.read()
            if frame is not None:
                images.append(frame)
                #cv2.imshow('Image', frame)
                #cv2.waitKey(1)
        else:
            
            cap.release()
            cv2.destroyAllWindows()
    initial_location, initial_frame = detect_ball_initial_location(images)
    centers, radii = track_ball(images, initial_location, initial_frame)
    