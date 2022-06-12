from cProfile import label
from curses import mousemask
from unittest import makeSuite
import cv2
import imutils
import pickle
import sklearn
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from trajectory import calculation, plot_animation, TRACK

class BallTracker:
    def __init__(self, images):
        self.orange_min = (5, 125, 125)
        self.orange_max = (10,255,255)
        self.images = images
        self.azimuth_regr = pickle.load(
                    open('models/azimuth_regr.sav', 'rb'))
        self.back_spin_regr = pickle.load(
                    open('models/back_spin_regr.sav', 'rb'))
        self.ball_speed_regr = pickle.load(
                    open('models/ball_speed_regr.sav', 'rb'))
        self.launch_angle_regr = pickle.load(
                    open('models/launch_angle_regr.sav', 'rb'))
        self.side_spin_regr = pickle.load(
                    open('models/side_spin_regr.sav', 'rb'))

    def detect_initial_location(self):
        n = 0
        for frame in self.images:
            x,y, radius = self.detect(frame)

            if radius is not None:
                if radius > 5:
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    
                    initial_location = (x,y)
                    initial_frame = n
                    print('Initial Location Detected')
                    self.determine_pix_to_meter_ratio(radius)
                    return initial_location, initial_frame
            n += 1
    
    def determine_pix_to_meter_ratio(self, r):
        ball_radius_pix = r
        ball_radius_ft = 0.07
        self.pix2ft = ball_radius_pix/ball_radius_ft
       

    def mask(self, frame):
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.orange_min, self.orange_max)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        return mask, cnts

    def detect(self, frame):
        # TODO: crop the frame around the ball adjust for ball size for faster inference
        mask, cnts = self.mask(frame)
        center = None
        radius = None
        x = None
        y = None
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)

            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            x = center[0]
            y = center[1]
        else:
            radius = self.last_r
            x = self.last_x
            y = self.last_y

        self.last_x = x
        self.last_y = y
        self.last_r = radius

        return x,y, radius 

    def track(self):

        def dist(location, proposed_location):
            return ((location[0]-proposed_location[0])**2 + (location[1]-proposed_location[1])**2)**0.5

        self.centers, self.radii = [], []
        self.initial_location, self.initial_frame = self.detect_initial_location()
        self.current_location = self.initial_location
        cv2.namedWindow('Tracking')
        for frame in self.images[self.initial_frame:]:
            x,y, radius = self.detect(frame)
     
            if radius > 15:
                if dist(self.current_location, (x,y)) <= 1000:
                    self.centers.append((x,y))
                    self.radii.append(radius)
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    cv2.circle(frame, (int(x), int(y)), int(radius),
                        (0, 255, 255), 2)
                    cv2.circle(frame, (x,y), 5, (0, 0, 255), -1)
                    cv2.waitKey(1)
                    cv2.imshow('Tracking', frame)
            else:
                self.radii.append(self.radii[-1])
                self.centers.append(self.centers[-1])

        self.detect_and_trim_initial_contact()
        shot_info = self.approximate_parameters()

    def detect_and_trim_initial_contact(self):
        max_delta_x = 0
        # TODO : only detect within smaller 
        for i in range(len(self.centers)):
            x = self.centers[i][0]
            y  = self.centers[i][1]
            r = self.radii[i]
            if i == 0:
                last_x = x
                last_r = r
                last_y = y

            delta_x = last_x - x
            delta_y = last_y - y
            delta_r = last_r - r

            if delta_x < -20:
                index = i
                break
        self.centers = self.centers[index:]
        self.radii = self.radii[index:]

    def convert_pix2fps(self, x_vel, y_vel, z_vel):
        fps = 120
        mph = 0.68
        return x_vel*self.pix2ft/fps/mph, y_vel*self.pix2ft/fps/mph, z_vel*self.pix2ft/mph
        

    def approximate_parameters(self):
        x = np.array([c[0] for c in self.centers])
        y = np.array([c[1] for c in self.centers])
        r = np.array(self.radii)
        x_vel = np.polyfit(np.arange(0,len(x)), x, 1)
        y_vel = np.polyfit(np.arange(0,len(y)), y, 1)
        z_vel = np.polyfit(np.arange(0,len(r)), r, 1)
        y_vel, z_vel, x_vel = self.convert_pix2fps(x_vel, y_vel, z_vel)
        shot_parameters = [x_vel[0], y_vel[0], z_vel[0]]
        print(shot_parameters)
        #shot_coeffs = [abs(s) for s in shot_parameters]
        shot_coeffs = shot_parameters
        azimuth = self.azimuth_regr.predict(shot_coeffs[0].reshape(-1, 1))[0][0]
        launch_angle = self.launch_angle_regr.predict(shot_coeffs[1].reshape(-1, 1))[0][0]
        ball_speed = self.ball_speed_regr.predict(shot_coeffs[2].reshape(-1, 1))[0][0]
        back_spin = self.back_spin_regr.predict(np.array(launch_angle).reshape(-1, 1))[0]
        side_spin = self.side_spin_regr.predict(np.array(azimuth).reshape(-1, 1))[0]
        shot_dict = {
            'ball_speed': ball_speed, 
            'launch_angle': launch_angle,
            'azimuth': azimuth, 
            'side_spin': side_spin/2, 
            'back_spin': back_spin
            }
        df = pd.DataFrame.from_dict(shot_dict, orient='index').transpose()
        print(df)
        ball = calculation(df)
        row1 = ball.iloc[0]
        result = TRACK(0.04593, 0.04267, 1.2, 0.16, 0.1,
                        row1[0], row1[1], row1[2], row1[5], row1[6], 0)
        x_track = result[0]
        y_track = result[1]
        z_track = result[2]
        coordinate_x = result[3]
        coordinate_y = result[4]
        coordinate_z = result[5]
        plot_animation(x_track, y_track, z_track)
        return shot_dict

    """
    def get_shot_coefficents(self):
        shot_coeffs = []
        flight_length = 0
        resid_thresh = {'x':.04, 'y':.2, 'width':.3}

        for col_cat in ['x', 'y', 'width']:
            frame_nos = flight_df.frame_no.values
            values = flight_df[col_cat].values / flight_df[col_cat].iloc[0]
            values = np.log(values)
            if col_cat == 'width':
                frame_nos = frame_nos[2:]
                values = values[2:]
            regr = LinearRegression().fit(frame_nos.reshape(-1, 1), values.reshape(-1, 1))

            y_pred = regr.predict(frame_nos.reshape(-1, 1))
            resids = abs(y_pred.reshape(-1) - values)
            while (max(resids) > resid_thresh[col_cat]):
                frame_nos = frame_nos[resids != max(resids)]
                values = values[resids != max(resids)]
                regr = LinearRegression().fit(frame_nos.reshape(-1, 1), values.reshape(-1, 1))
                y_pred = regr.predict(frame_nos.reshape(-1, 1))
                resids = abs(y_pred.reshape(-1) - values)
            flight_length = len(frame_nos)
            shot_coeffs.append(regr.coef_[0][0])
        shot_coeffs = np.array(shot_coeffs)

        azimuth = self.azimuth_regr.predict(shot_coeffs[0].reshape(-1, 1))[0][0]
        launch_angle = self.launch_angle_regr.predict(shot_coeffs[1].reshape(-1, 1))[0][0]
        ball_speed = self.ball_speed_regr.predict(shot_coeffs[2].reshape(-1, 1))[0][0]

        back_spin = self.back_spin_regr.predict(np.array(launch_angle).reshape(-1, 1))[0]
        side_spin = self.side_spin_regr.predict(np.array(azimuth).reshape(-1, 1))[0]


        return {'ball_speed': ball_speed*0.882, 'launch_angle': launch_angle,
                'azimuth': azimuth, 'side_spin': side_spin/2, 'back_spin': back_spin},\
               shot_coeffs

    """

if __name__ == "__main__":
    images = []
    test_file='tmp/testvid5.mp4'
    cap = cv2.VideoCapture(test_file)

    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    print('Video opened. Reading images. ')
    print(test_file)
    ret, frame = cap.read()
    while(cap.isOpened()):
        if ret:
            ret, frame = cap.read()
            if frame is not None:
                images.append(frame)
        else:
            
            cap.release()
            cv2.destroyAllWindows()

    print('Initiating Tracker')
    tracker = BallTracker(images)
    print('Tracking')
    tracker.track()
    #plt.plot(tracker.radii, label='radii')
    #plt.plot([c[0] for c in tracker.centers], label='x')
    #plt.plot([c[1] for c in tracker.centers], label='y')
    #plt.legend()
    #plt.savefig('trajectory.png')