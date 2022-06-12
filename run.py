import os
import re
import math
import pickle
import numpy as np
from datetime import datetime, timedelta

import cv2 
import os 

from predict import Step2
import make_traj


def extract_factors(video_path: str) -> dict:
    """
    Extract the five factors from the video for trajectory prediction.
    Five factors are: ball speed, launch angle, azimuth, side-spin, back-spin

    :param video_path: local file path of the original video.
    :return: five factors in a dictionary {factor_name: value}
    """
    # load prerequisites
    impact_finder = pickle.load(
        open('models/impact_finder.sav', 'rb'))
    classify_bbox = pickle.load(
        open('models/classify_bbox_new.sav', 'rb'))
    azimuth_regr = pickle.load(
        open('models/azimuth_regr.sav', 'rb'))
    back_spin_regr = pickle.load(
        open('models/back_spin_regr.sav', 'rb'))
    ball_speed_regr = pickle.load(
        open('models/ball_speed_regr.sav', 'rb'))
    launch_angle_regr = pickle.load(
        open('models/launch_angle_regr.sav', 'rb'))
    side_spin_regr = pickle.load(
        open('models/side_spin_regr.sav', 'rb'))

    # hard-code ball_init for now
#     ball_init = [357, 781, 25, 22]

    # process raw video and extract five input features -> dict
    s2 = Step2()
    factors, f2 = s2.get_shot_factors(video_path, impact_finder, classify_bbox,
                                      azimuth_regr, back_spin_regr,
                                      ball_speed_regr, launch_angle_regr,
                                      side_spin_regr)

    if factors["ball_speed"] < 70:
        factors["ball_speed"] = 70.23
    if factors["ball_speed"] > 120:
        factors["ball_speed"] = 120.12

    if factors["launch_angle"] < 15:
        factors["launch_angle"] = 15.13
    if factors["launch_angle"] > 40:
        factors["launch_angle"] = 40.54

    if factors["azimuth"] < -15:
        factors["azimuth"] = -15.45
    if factors["azimuth"] > 15:
        factors["azimuth"] = 15.17

    if factors["side_spin"] < -2000:
        factors["side_spin"] = -1986.87
    if factors["side_spin"] > 2000:
        factors["side_spin"] = 2045.91

    if factors["back_spin"] < 1000:
        factors["back_spin"] = 1005.98
    if factors["back_spin"] > 7500:
        factors["back_spin"] = 7472.67

    return factors


def predict(factors: dict, gif_name: str) -> tuple:
    """
    Make trajectory prediction.

    :param factors: five factors in dictionary format
    :param gif_name: the file name for output gif file
    :param is_auth: check if the user has logged in
    :return: carry(x1), offline(y) and total distance(x2) of the shot
    """
    simulation = make_traj.TrackSimulation()

    reg_carry = pickle.load(
        open('models/reg_carry.pkl', 'rb'))
    reg_tot_dist = pickle.load(
        open('models/reg_tot_dist.pkl', 'rb'))
    reg_offline_ratio = pickle.load(
        open('models/reg_offline_ratio.pkl', 'rb'))
    reg_peak_height = pickle.load(
        open('models/reg_peak_height.pkl', 'rb'))

    x1, y, x2, track = simulation.traj(factors, reg_carry, reg_tot_dist,
                                       reg_peak_height, reg_offline_ratio)

    # simulation.make_anime(track, gif_name)
    simulation.make_plotly(track, gif_name)

    return x1, y, x2

def create_video(input_cam=0):
    cv2.namedWindow("Camera ", input_cam)
    vc = cv2.VideoCapture(input_cam)
    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
    #os.mkdir('tmp', exist_ok=True)
    frame_num= 0 
    img_array = []
    while rval:
        rval, frame = vc.read()
        size = frame.shape[:-1]
        img_array.append(frame)
        key = cv2.waitKey(0)
        if key == 27: # exit on ESC
            break

    out = cv2.VideoWriter('tmp/last_shot.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def shot(input_cam=0):
    user_input = input('Press any key to start the video: ')
    if user_input:
        print('Starting video. Take your shot. ')
        create_video()
        factors = extract_factors('tmp/shot_12.m4v')
        prediction = predict(factors, 'result.gif')
        
if __name__=="__main__":
    shot()