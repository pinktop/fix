## TemProject of Computer Vision Class
2021년 세종대학교 Computer Vision 수업의 Temproject Challenge를 위한 소개 Repository 입니다.

## Challenge Overview
[챌린지 소개영상](https://youtu.be/phcP6AtCtyc)

[해당 분야 서베이 결과](https://github.com/hyj378/-TemProject-2021ComputerVision/files/7366160/default.pdf)

본 챌린지는 Semi-Supervised Learning을 통한 Image Classification을 진행합니다. baseline method는 [FixMatch](https://arxiv.org/abs/2001.07685)를 이용하였으며 [Link](https://github.com/kekmodel/FixMatch-pytorch)를 baseline code로 사용하였습니다.

해당 분야 서베이로는 unlabeled dataset에서 효과적으로 학습에 사용할 데이터셋을 선정하는 방법인 Active Learning과 2021년도 CVPR에 공개된 Meta Pseudo Labels 연구를 서베이 하였으며 링크를 올려놓았습니다.

- **Image Classification**

  Image Classification이란[1] 입력된 이미지를 미리 정해진 카테고리 중 하나로 분류하는 작업으로 Computer Vision의 core problems 중 하나입니다. 이는 가장 간단한 태스크이면서 실제로 활용될 수 있는 활용 범위가 넓은 과제입니다. Image Classification은 [컴퓨터 비전 2주차 수업](https://youtu.be/Q44g-lZwjzU)에서도 자세하게 소개되었습니다. 따라서 Semi-supervised Learning에 대한 설명으로 바로 가시려면 아래로 이동해주세요.
  Image Classification model은 이미지를 입력으로 받으며, RGB 이미지의 경우 0에서 255사이의 상수 메트릭스를 (R,G,B) 3 채널로 구성됩니다. 즉 이 수많은 숫자를 하나의 label로 분류하는 과정입니다.
  ![image](https://user-images.githubusercontent.com/41140561/137722145-f6ee30bd-1228-4064-9366-f8dee2395d64.png)

  
- **Semi-supervised Learning**

  머신러닝을 Supervised Learning(지도학습), Unsupervised Learning(비지도학습), Reinforcement Learing(강화학습)의 세가지 학습 방법으로 나누는 것에 대해서는 많이 보셨을겁니다. 간단하게 소개드리자면 다음과 같습니다.[2]
  - Supervised Learning
 
      dataset points {x<sup>(1)</sup>, ..., x<sup>(m)</sup>}과 그에 대한 결과값 {y<sup>(1)</sup>, ..., y<sup>(m)</sup>}가 주어지고 x로부터 y를 예측하는 분류기를 설계하는 학습방법.
    
  - Unsupervised Learning
  
      unlabeled dataset points {x<sup>(1)</sup>, ..., x<sup>(m)</sup>}에서 hidden patterns을 찾는 학습 방법.
      
  - Reinforcement Learning
 
      기존 모델이 loss를 통해 모델의 가중치와 편향을 학습하는것과 다르게 보상이라는 개념을 통해 모델을 학습하는 방법.

  Machine Learning 기법[3]이 발전하면서 Supervised Learning 분야에서는 Data이 부족이 큰 bottleneck이 되었으며. 이러한 현상을 해결하기 위해 데이터가 없어도 지속적으로 학습할 수 있는 Weak Supervision 방식의 접근법이 연구되고있으며, Semi-Supervised Learning도 Weak Supervision의 접근방식 중 하나입니다.
  
  ![image](https://user-images.githubusercontent.com/41140561/137734258-2a2f3d16-2c7e-4ad2-9547-5da7a257619d.png)


- **Dataset (CIFAR-10)**
  CIFAR-10 dataset은 10개의 Class로 구성된 데이터셋으로 32x32 해상도의 이미지가 각 클래스당 6000개로 구성되어 있습니다. 50000개의 Train Data와 10000개의 Test Data로 구성되어있으나, 학습시에는 40개의 이미지만을 이용하여 주세요.
  
  ![image](https://user-images.githubusercontent.com/41140561/137735550-1ba008b7-c52a-4609-b82f-51335c1d54a0.png)


## result 파일 제출 방법
위에서 소개 드린 baseline github를 이용하셨다면, 해당 체크포인트로 result.py를 통해 제출할 json 파일을 생성하실 수 있습니다.
제출 형식에 관한 정보는 [Submission Guidelines](http://203.250.148.129:3088/web/challenges/challenge-page/31/submission)에서 확인하실 수 있습니다.
```bash
python result.py --arch wideresnet --batch-size 64 --seed 5 --resume {체크포인트 위치} --save {저장할 json 위치}
# 예시
# python result.py --arch wideresnet --batch-size 64 --seed 5 --resume checkpoint/model_best.pth.tar --save here.json
```

## Caution
- 학습 시 CIFAR-10 데이터 기준으로 하나의 class 당 4개의 labeled set을 사용하여 총 **40개의 labeled dataset을 이용하여** 학습합니다.

## 참고자료
- [1] Image Classification 설명: [CS231n](https://cs231n.github.io/classification/)
- [2] Machine Learning 설명: [CS229](https://stanford.edu/~shervine/teaching/cs-229/)
- [3] Weak Supervision: [Weak Supervision: The New Programming Paradigm for Machine Learning](https://dawn.cs.stanford.edu/2017/07/16/weak-supervision/)
- [4] CIFAR-10 dataset: [Link](https://www.cs.toronto.edu/~kriz/cifar.html)
