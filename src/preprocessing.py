import sys
import os
import json
import io
import numpy as np
import operator
import collections
import base64
import pandas as pd
from glob import glob
from IPython.display import HTML
import ray
import matplotlib.pyplot as plt
import dlib

import logging

import cv2
from utils.inference import load_image
from utils.inference import load_detection_model
from utils.inference import detect_faces
from utils.inference import apply_offsets

emotion_offsets = (20, 40)
emotion_offsets = (0, 0)

detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_alt2.xml'

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# multi-modal 데이터셋 다운로드 위치 입력

# 입력 zip파일의 part 번호
part_num = sys.argv[1]

# input 폴더
base_folder = '/tf/notebooks/datasets/emotion/multi-modal/part'+part_num+'/'

# output 파일경로
prep_trainset_path='../datasets/kor_multi_modal/prep_part'+part_num+'.csv'
prep_trainset_frontal_face_path='../datasets/kor_multi_modal/prep_part'+part_num+'_frontal_face.csv'

ray.init()
face_detect = dlib.get_frontal_face_detector()

# 맵핑
def get_tagging_info(json_loaded, img_path_list):
    train_data = []
    for shot_info_idx in range(len(json_loaded["shot_infos"])):
        for visual_info_idx in range(len(json_loaded["shot_infos"][shot_info_idx]["visual_infos"])):
            img_path = json_loaded["shot_infos"][shot_info_idx]["visual_infos"][visual_info_idx]["image_id"]
            if json_loaded["shot_infos"][shot_info_idx]["visual_infos"][visual_info_idx]["persons"] != []:
                for person_info_idx in range(len(json_loaded["shot_infos"][shot_info_idx]["visual_infos"][visual_info_idx]["persons"])):
                    if json_loaded["shot_infos"][shot_info_idx]["visual_infos"][visual_info_idx]["persons"][person_info_idx]["person_info"]["emotion"] != []:
                        person = json_loaded["shot_infos"][shot_info_idx]["visual_infos"][visual_info_idx]["persons"][person_info_idx]
                        face_rect = person['person_info']['face_rect']
                        emotion = person["person_info"]["emotion"]

                        # 수치가 가장 큰 감정 선택
                        max_emo = max(emotion.items(), key=operator.itemgetter(1))[0]
                        
                        img_exist_path = [s for s in img_path_list if img_path in s]
                        if img_exist_path == []:
                            continue
                        
                        full_img_path = img_exist_path[0]
                        
                        if full_img_path != []:
                            
                            # face_rect 부분 잘라내기
                            # ex) 'max_x': 1125, 'max_y': 798, 'min_x': 708, 'min_y': 267
                            min_x = face_rect['min_x']
                            max_x = face_rect['max_x']
                            min_y = face_rect['min_y']
                            max_y = face_rect['max_y']

                            if  not isinstance(min_x, int) or not isinstance(max_x, int) or  not isinstance(min_y, int) or not isinstance(max_y, int) or \
                                min_x <= 0 or max_x <= 0 or min_y <= 0 or max_y <= 0:
                                continue

                            # haar face 디텍션해서 얼굴 잡히는 데이터만 사용 (20190720추가)
                            img = cv2.imread(full_img_path)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            img = img[min_y:max_y, min_x:max_x]
#                             logger.debug("read img")
                            
#                             gray_image = load_image(full_img_path, grayscale=True, color_mode="grayscale")
#                             gray_image = np.squeeze(gray_image)
#                             gray_image = gray_image.astype('uint8')

                            # haar face 디텍션해서 얼굴 잡히는 데이터만 사용 (20190720추가)
#                             faces = cv2.CascadeClassifier(detection_model_path).detectMultiScale(img, 1.3, 5) 
                            faces = face_detect(img)

                            
                            if len(faces) == 0:
                                continue
                            else:
                                train_data.append({'emotion':max_emo, 'img_path':full_img_path, 'face_rect':face_rect})
                                        
            
    return train_data

@ray.remote
def get_raw_tag_info_from_tagging_info(video_clip_path):
    # 해당 비디오 클립의 전체 이미지 경로
    img_path_list = glob(video_clip_path+"/*/*/*.jpg")    

    # 해당 비디오 클립의 태깅정보 파일 경로
    json_file_name_postfix = video_clip_path.split("_")[-2]+"_"+video_clip_path.split("_")[-1]
    video_clip_tagging_json_path = glob(os.path.join(video_clip_path, "*"+json_file_name_postfix+"_interpolation.json"))

    with open(video_clip_tagging_json_path[0], 'r') as f:
        json_loaded = json.load(f)
    
    return get_tagging_info(json_loaded, img_path_list)

video_clip_pathes = glob(base_folder+"/*")

tmp_raw_tag_info = ray.get([ get_raw_tag_info_from_tagging_info.remote(video_clip_path) for video_clip_path in video_clip_pathes ])

raw_tag_info = []

for tag_info_list in tmp_raw_tag_info:
    raw_tag_info.extend(tag_info_list)

raw_tag_info_df = pd.DataFrame(raw_tag_info)
raw_tag_info_df.to_csv(prep_trainset_frontal_face_path, index=False)

ray.shutdown()
