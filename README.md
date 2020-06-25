# korean-image-sentiment-analysis
This project is a part of the [mercy-project](https://github.com/mercy-project).

#### -- Project Status: [Active, On-Hold, Completed]

## Project Intro/Objective
이 프로젝트의 목적은 딥러닝 컴퓨터 비전 기술을 이용해서 사진에서 사람의 감정을 추출하는 것입니다.

### Methods Used
* OpenCV Haar cascade
* Dlib HOG(Histogram of Oriented Gradients) based
* Simple version of XCEPTION (https://github.com/oarriaga/face_classification)

### Technologies
* tf.keras 2.x

## Project Description
데이터 소스: AIHUB(http://www.aihub.or.kr/)

### step 1: Image frame 단위 raw 데이터 로딩 학습 dataset file 생성
* inputs:
    * 태깅파일: ~_interpolation.json
    * 이미지파일: ~KM_0000000000.jpg
    * 감정종류: 8가지 ('happiness', 'afraid', 'neutral', 'surprise', 'sadness', 'contempt', 'anger', 'disgust')
        * 예시: {'happiness': 0, 'afraid': 0, 'neutral': 10, 'surprise': 0, 'sadness': 0, 'contempt': 0, 'anger': 0, 'disgust': 0} 일 경우 'neutral'이 선정
* 최종 CSV dataset file 생성
    * 태깅데이터에서 얼굴 검출 되는 이미지만 추출    

### step2: Model 학습
    1. face detection
    2. emotion detection
    3. tflite 변환 테스팅
    
### 1. face detection

* OpenCV Haar cascade 방식
    * 사용이유: CPU 상 검출 시간이 빠르고, 적용하기 용이함
    * 버전: haarcascade_frontalface_alt2
    * 설명 내용 출처: https://thebook.io/006939/ch13/02/
![](https://thebook.io/img/006939/p412.jpg)
     캐스케이드 구조 1단계에서는 얼굴 검출에 가장 유용한 유사-하르 필터 하나를 사용하여, 얼굴이 아니라고 판단되면 이후의 유사-하르 필터 계산은 수행하지 않습니다. 1단계를 통과하면 2단계에서 유사-하르 필터 다섯 개를 사용하여 얼굴이 아닌지를 검사하고, 얼굴이 아니라고 판단되면 이후 단계의 검사는 수행하지 않습니다.
<img src='https://thebook.io/img/006939/p411.jpg' width=300 />
<center>얼굴 검출에 유용한 유사-하르 필터의 예</center>

* Dlib HOG(Histogram of Oriented Gradients) based 방식
    * 사용이유: CPU 상 OpenCV Haar cascade 방식보다 성능이 좋음, 적용하기 용이
    * 설명 내용 출처: https://medium.com/@jongdae.lim/기계-학습-machine-learning-은-즐겁다-part-4-63ed781eee3c
    * 검출방식 비교: https://seongkyun.github.io/study/2019/03/25/face_detection/

<img src='https://miro.medium.com/max/1022/1*dP0Ixs4vHGUKCScufH9_Vw.jpeg' width=300 />
<center>1) 이미지를 흑백으로 변환</center>
<img src='https://miro.medium.com/max/1600/1*lsNRg_1oOELFcug_AjlkqQ.gif' width=300 />
<center>2) 이미지를 16x16 픽셀의 작은 정사각형들로 분해 한 후 각 정사각형에서, 그래디언트가 주요 방향(윗쪽, 우상쪽, 오른쪽, 등)을 얼마나 가리키고 있는지 카운팅합니다.</center>
<center>그런 다음 이 사각형을 가장 강한 화살표 방향을 나타내는 벡터로 변환 합니다.</center>
<img src='https://miro.medium.com/max/1600/1*HtgQZ4guaIo8wflbsR1MLw.png' width=500 />

### 2. emotion detection
* mini_XCEPTION 모델
* 사용이유: 인식시간이 빠르고, 모델이 가볍기에 온디바이스에 적용가능
* 학습내용:
    * 학습데이터: AIhub 멀티모달 데이터(http://www.aihub.or.kr/content/555)로 학습
        * 클래스 밸런싱된 48 x 48 gray scale image 128,499만여개에서 샘플링하여 사용
    * 학습된 모델을 tflite로 변환
* 모델출처: https://github.com/oarriaga/face_classification


## Getting Started
1. Install packages
```shell
python >= 3.6
pandas
ipywidgets
matplotlib
opencv-python
statistics
h5py
numpy
tensorflow-gpu
ray # for preprocessing
```
2. Download Raw data from AIhub(http://www.aihub.or.kr/content/555)
3. Preprocessing with [codes](src/00_vision_model_total_process.ipynb)
4. Training the model
```shell
$ python src/train_emotion_kor_multi_modal_classifier.py
```
5. Inference


## Contributing Members
|Name     |  Slack Handle   | 
|---------|-----------------|
|[Full Name](https://github.com/[github handle])| @johnDoe        |
|[Full Name](https://github.com/[github handle]) |     @janeDoe    |
