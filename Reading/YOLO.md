## You Only Look Once: Unified, Real-Time Object Detection

논문 링크 : https://arxiv.org/pdf/1506.02640.pdf

-------------------

### Abstract
object dtection을 공간적으로 분리된 bounding boxes와 클래스 확률로 regression을 간주함.

single-neural network는 한 번의 평가로 전체 이미지에서 bounding boxes와 클래스 확률을 예측함.

전체 detection pipeline이 single-neural network이기 때문에, detection 성능에 대해 직접적으로 end-to-end로 최적화될 수 있으며 이러한 구조는 매우 빠르게 진행 됨.

YOLO는 localization error를 더 많이 만들지만, 배경에 대한 false positive으로 예측 가능성은 낮음. 

결과적으로 객체의 일반적인 표현을 학습한다고 말할 수 있음.

사물의 일반적 표현을 다른 도메인으로 일반화할 때 DPM(deformable parts models)과 R-CNN을 포함한 다른 탐지 방법들 능가함.

### Introduction
현재의 detection system은 분류기에 detection 역할을 하도록 재설정하는 방식으로 객체를 탐지하기 위해서, 시스템은 객체를 분류기가 받고 테스트 이미지 내에서 다양한 위치와 크기를 평가함.

DPM(deformable parts models)같은 경우는 sliding window를 사용하여 분류기가 전체 이미지에 대해 모든 공간에서 균등하게 실행되도록 함.
R-CNN의 경우는 region proposal의 방법을 사용하여 먼저 이미지 내에 잠재적인 bounding boxes를 생성하고 그 제안된 boxes에 대해 분류기를 실행하여 분류 이후 bounding boxes를 정교하게 하고, 중복되는 detection을 제거하여 다른 객체에 기반해서 boxes를 재점수함. - 이러한 방식은 복잡한 파이프라인은 개개인의 요소가 분리되어서 학습되어야 하기 때문에 연산량이 많고 최적화 하는데 어려움.

object detection을 단일 regression 문제로 재구성하고, 이미지 픽셀에서 bounding box 좌표와 클래스 확률까지 쭉 이어지도록 하였다. - 그래서 YOLO(You Only Look Once)라고 명칭함. - "객체가 무엇이고 어디 있는지를 예측하기 위해 이미지를 '단 한 번만 본다.'"

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81404373-477d9a80-9170-11ea-9f92-b1d6d10bd90f.png" width="75%"></p>

The YOLO Detection System.

    - 1. 먼저 입력이미지를 448 x 448의 크기로 조정한다.
    - 2. 이미지에 대해 단일 Convolution Network를 실행함.
    - 3. NMS(Non-Max Suppresion - 중복된 detection을 제거) - 모델의 신뢰도에 의한 결과 detection의 임계값을 정함.
  
간단하게 말하자면 단일의 Convolutional Network는 동시에 복수의 Bounding boxes를 예측하고, boxes에 대한 클래스 확률을 예측함.
따라서 전체 이미지에 대해 학습하고 detection 성능을 직접적으로 최적화함.

YOLO의 장점 3가지.

1. 매우 빠름.

detection을 regression 문제로 재구성하여 복잡한 파이프라인이 필요하지 않게 됨.

Titan X GPU에서 배치 없이 45fps(frame per second), 빠른 버전 150fps보다 빠름. - real time으로 적용할 수 있다는 것을 의미, 다른 real time 시스템에 비해 두 배가 넘는 mAP를 달성함.

2. 이미지에 대해 전체적으로 추론함.

Sliding window와 region proposal 기반의 기술과 다르게, YOLO은 학습과 테스트시에 전체 이미지를 보기 때문에, 명백하게 그들의 외관과 같은 클래스에 대한 문맥상의 정보를 encodes한다.

YOLO는 Fast R-CNN과 비교해서, 절반 이하의 배경 오류를 발생 시킴.

3. 객체의 일반적인 표현을 학습함.

자연 이미지와 예술 이미지 학습하였을 때, YOLO는 DPM과 R-CNN보다 높은 성능을 얻을 수 있었음.
또한 매우 일반적이여서 새로운 도메인이나 입력에 대해서도 실패할 가능성이 적음.

### Unified Detection
전체 이미지로부터의 특성으로 각 bounding box를 모든 클래스에 대해 동시에 예측함. - 모든 객체를 전체적으로 잘 추론하고 있다는 것을 의미.

따라서 높은 AP(Average Precision)을 유지하며 end-to-end로 학습되고 real time속도를 가능하게 함.

동작 방식으로는 입력 이미지를 S x S grid로 나눠서 객체의 중심에 grid cell이 들어갔다면, 그 객체를 탐지하는 역할을 함.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81413953-3f7a2680-9181-11ea-9f98-bffa77d36696.png" width="75%"></p>

각 grid cell은 bounding boxes B와 boxes에 대한 신뢰 점수를 예측함. - 신뢰 점수는 box가 객체를 보유하고 있다고 생각하는 모델의 신뢰도와 예측하는 box의 정확도를 반영, <img width="124" alt="스크린샷 2020-05-08 23 11 54" src="https://user-images.githubusercontent.com/45933225/81414002-4f920600-9181-11ea-8101-e8cc9f7af219.png"> 으로 표현.
.

각 bounding box는 5개의 예측으로 구성됨. - x, y, w, h 그리고 신뢰도이다.
        
        - (x, y)는 grid cell의 경계를 기준으로 box의 중심좌표를 나타냄.
        - 너비와 높이는 전체 이미지에 대해 예측함.
        - 신뢰도 예측은 예측된 box와 어느 ground truth box 사이의 IOU를 나타냄.

각 grid cell은 조건부 클래스 확률인 C를 <img width="130" alt="스크린샷 2020-05-08 23 45 18" src="https://user-images.githubusercontent.com/45933225/81417320-fa0c2800-9185-11ea-8351-6e05bb6557df.png"> 예측함. - 여기서 확률은 객체를 보유하고 있는 grid cell에 대한 조건부를 나타냄.
boxes의 갯수인 B에 상관하지 않고, 오직 grid cell당 하나의 클래스 확률을 예측함. - 테스트시에는 조건부 확률과 개개인의 신뢰도 예측을 곱하면 위에 합쳐져 있는 식이 나옴.

결과적으로 각 box에 대해 클래스별 신뢰도를 알려줌. - 점수는 box에서 클래스가 나타나는 확률과 예측된 box가 객체와 얼마나 잘 맞는지를 담고 있다.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81416879-5458b900-9185-11ea-96d5-3037822b9359.png" width="75%"></p>

시스템은 detection regressopm 문제로 모델링한 후 이미지를 S × S grid와 각 gird로 나눔.
grid cell은 B 경계 상자, 해당 상자에 대한 신뢰도를 예측하여 C 클래스 확률을 구함. - 이러한 예측은 다음과 같이 인코딩되어 있음(S × S × (B ∗ 5 + C) tensor).

Pascal VOC에 대해서는 S = 7, B = 2를 사용하고 클래스가 20개 이기에 C = 20이어서 최종 예측은 7 X 7 X 30 tensor가 됨.

#### Network Design
Fully connected layers가 출력의 확률과 좌표를 예측할 때, 네트워크의 초반 Convolutional Layers는 이미지로부터 특성을 추출함.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81416602-e57b6000-9184-11ea-935b-aa427c8f4aeb.png" width="75%"></p>

YOLO Network 구조는 이미지 분류 모델인 GoogLeNet 모델로부터 영감을 얻음.

YOLO Network는 24개의 Convolution Layer와 2개의 FC Layer로 이루어져 있음.

다른점으로는 GoogLeNet에서 인셉션 모듈을 썻던 것 대신, YOLO는 1 X 1 Reduction Layers를 사용하고, 그 뒤에 3 X 3 Convolution Layer을 사용함.

추가적으로 Fast YOLO는 더 적은 Convolution Layer(Convolution Layer 24개 -> 9개)와 더 적은 filter를 사용하였음. - YOLO와 Fast YOLO는 네트워크 크기를 제외하고 학습, 테스트 파라미터는 동일함.

#### Training
Convolution Layer들은 ImageNet 1000 Class에 대해 pretrain 함. pretraining을 위해 처음 20개의 Convolution Layer들을 사용하고, 그 뒤에 Average Pooling과 FC Layer을 사용함.

그다음 모델을 detection 역할을 하도록 변환하여 이후에 4개의 Convolution Layer와 2개의 FC Layer를 추가함.

detection에는 종종 세밀한 시각 정보가 필요하기 때문에 네트워크의 입력 해상도를 224 X 224에서 448 X 448으로 높임.

마지막 Layer는 클래스 확률과 bounding box 좌표를 예측함.

이미지의 너비와 폭을 기준으로 bounding box 너비와 높이를 0과 1 사이에 맞게 정규화 함.
그리고 bounding box의 좌표 (x, y)는 특정 grid cell 위치의 offsets값을 사용하여 0과 1 사이에 오게 함.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81420951-aef51380-918b-11ea-86c7-6308d1776f6a.png" width="50%"></p>

마지막 Layer에 선형 활성화 함수를 사용하고, 다른 모든 Layer는 낮게 조정된 선형 활성화 함수를 사용함.

모델의 출력의 SSE(Sum Squared Error)를 최적화 함. - AP를 최대화 하는데에는 최적화는 아님. 이것은 이상적이지 않은 분류 오류와 localization 오류에 동등하게 가중치를 부여함. 또한, 모든 이미지의 많은 grid cells는 객체도 포함하고 있지 않음.

따라서 그러한 cells의 '신뢰' 점수를 0으로 향하게 하며, 가끔 객체를 포함하는 cells의 gradient를 못쓰게 만듬. - 이것은 모델의 불안정성을 유발하여, 학습이 초기에 분산하게 함.

이것을 해결하고자 bounding box 좌표 예측의 loss를 증가시키고, 객체를 포함하지 않는 boxes에 대한 예측 신뢰도로부터의 loss를 줄여야 함.
그래서 다음과 같은 lambda함수를 사용<img width="175" alt="스크린샷 2020-05-12 13 39 58" src="https://user-images.githubusercontent.com/45933225/81639147-1514bb80-9456-11ea-88f8-a8a3665a0caa.png">.


SSE는 또한 큰 박스거나 작은 박스의 오류에 대해 동등하게 가중치를 줌. 여기서 error metric은 큰 박스에서의 작은 편차가 작은 박스에서 작은 편차보다 덜 중요하다는 것을 반영해야 함.

이를 부분적으로 다루기 위해 bounding box의 너비와 높이의 제곱근을 예측함.

YOLO는 grid cell당 여러 bounding boxes를 예측함. 학습 진행중에는 각 객체에 대해 bounding box predictor를 원함.

따라서 하나의 predictor에 객체를 예측하는 것을 중요시 함. - 예측이 ground truth와 높은 IOU를 가지는 것에 기반

그래서 bounding box  predictors간에 전문화로 이어짐으로써 각 predictor는 특정 크기, 종횡비, 또는 객체의 클래스를 잘 예측하여 전체적인 recall(재현율)을 개선시킴.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81639638-76895a00-9457-11ea-89ca-a677f88248bd.png" width="50%"></p>

학습 시에는 다음의 multi-part 비용 함수를 최적화 함.

Loss function이 객체가 grid cell안에 있을 때, 오직 분류 에러에만 패널티를 줌. - 조건부 클래스 확률

또한 predictor가 ground truth box에 대해 책임이 있을 때, bounding box error에 패널티를 줌. - grid cell에서 predictordml IOU가 가장 높을 경우

- network training

        train, validation dataset - VOC 2007, 2012
        epochs 135
        batch 64, momentum 0.9, decay 0.0005
        learning rate 0.001 -> 0.01 천천히 상승. -  높은 lr사용 시 gradients 발산.
        0 ~ 30 epochs 0.0001, 30 ~ 75 epochs 0.001, 75 ~ 135 epochs 0.01
        overfitting dropout 0.5, data augmentation 임의적 스케일링, 원본 이미지 크기의 최대 20%정도 사용, 임의적으로 이미지의 exposure과 saturation을 조정(HSV 색 공간의 1.5배율).
        
#### Inference
PASCAL VOC 이미지당 98개의 bounding boxes와 각 박스당 클래스 확률을 예측함.

grid design은 bounding box predictions 내의 공간적 다양성을 구현함. - 어떤 물체가 어떤 grid cell에 속하는지 명확하고 네트워크는 각 물체에 대해 하나의 box만을 예측함. 하지만, 커다란 객체나 여러 cells의 경계에 가까운 객체는 여러 cells에 의해 잘 localize 될 수 있음.
NMS(Network Management System) mAP를 2 ~ 3% 향상시킴.

#### Limitation of YOLO

    1. grid cell이 하나의 클래스만 예측하므로 작은 object가 주변에 있으면 제대로 예측하기 힘듬.
    2. 학습 데이터로부터 bounding box의 형태를 학습하므로, 새로운 형태의 bounding box의 경우 예측하기 어려움.
    3. Localization에 대한 부정확함.
    
### Comparison to Other Detection Systems
Detection 파이프라인은 일반적으로 입력 이미지들로부터 특정 셋을 추출하는데서 시작하여 분류기나 localizers 특정 공간에서 객체를 인식하는데 사용함. 이 분류기나 localizers는 전체 이미지의 regions 하위 세트에 대해 sliding window를 실행함.

#### Deformable parts models
sliding window 접근 방식을 사용하여 object detection을 함.

DPM은 분리된 파이프라인을 사용해서 정적인 특성을 추출하고, regions를 분류하고, 높은 점수를 가진 regions에 대해 bounding boxes를 예측 함. 하지만, YOLO는 이 종류가 다른 부분들을 단일의 CNN으로 대체한다.

YOLO의 통합된 구조는 DPM보다 빠르고, 더 정확하도록 만듬.

#### R-CNN
그 변형들은 이미지내의 객체를 찾기 위해 sliding windows 대신에 region proposals를 사용함.

- 복잡한 파이프라인

        Selective Search는 잠재적 bounding boxes를 생성 -> conv network로 특성 추출 -> SVM으로 boxes를 점수화 -> 선형 모델 bounding boxes를 조정 -> NMS로 중복된 detections를 제거

따라서 각 단계는 정교하게 독립적으로 조정되어야 하고, 결과 시스템은 테스트 시에 이미지당 40초가 넘게 걸릴 정도로 매우 느림.

그래도 YOLO는 R-CNN과 약간의 유사성을 공유함.

각 grid cell은 잠재적 bounding boxes를 제안하고, convolutional 특성을 이용해서 boxes를 점수 매김. 하지만, YOLO는 gride cell proposals에 공간적 제약을 가함으로써 다중 탐지를 완화시키며 R-CNN은 Selective Search가 이미지당 2,000개의 boxes를 제안하는데 비해, YOLO는 이미지당 98개만을 제안함. 그리고 개별적인 요소들을 단일의 공동 최적화 모델로 결합함.

#### Other Fast Detectors
Fast, Fast R-CNN은 연산을 공유하고 Selective Search 대신에 NN을 이용하여 regions를 제안하도록하여 R-CNN의 속도를 높이는데 초점을 맞춤.

Fast R-CNN, Faster R-CNN 이 두 가지 R-CNN의 모두 속도와 정확성에 대해 향상했지만, real-time 성능에는 아직 모자름.

YOLO는 커다란 detection 파이프라인의 개별 구성 요소를 최적화하려고 하는 대신 파이프라인을 없애고 설계상 신속하게 함. 그리고 다양한 물체를 동시에 감지하는 법을 배우는 '범용 검출기'라고 함.

#### Deep MultiBox
단일 object detection을 신뢰도 예측을 단일 클래스 예측으로 대체함으로 가능하게 함. 하지만, MultiBox는 보편적인 object detectio을 하지 못하고, 여전히 큰 파이프라인의 일부분에 불과하므로 추가적인 이미지 패치 분류가 필요함.

Deep MultiBox도 YOLO처럼 bounding box를 예측하기 위해 Convolutional Network를 사용함.
그에 반해, YOLO는 완전한 탐지 시스템을 갖춤.

#### OverFeat
CNN을 이용하여 localization을 하고 localizer를 적응시켜 탐지를 하도록 함.

localization을 위한 최적화이며 탐지로는 아니다. DPM처럼, 예측할 때에 localizer은 지역 정보만을 봄. 따라서 OverFeat은 전체적인 맥락에서 추론하지 못하고, 일관된 detection을 만들기 위해 상당한 후처리가 필요함.

#### MultiGrasp
하나의 객체를 보유하고 있는 이미지의 경우 파악 가능한 단일 영역만 예측하면 됨.

크기, 위치, 객체의 경계, 클래스를 예측하지 않고 지역만을 찾음으로써 이미지 내의 다양한 클래스의 다양한 객체에 대해 bounding boxes와 클래스 확률을 예측하는 YOLO와 차이점을 갖음.

### Experiments
PASCAL VOC 2007에 대해 비교함.

#### Comparison to Other Real-Time Systems

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81649674-a4789980-946b-11ea-801e-5e646dcb9dcb.png" width="50%"></p>

Fast YOLO는 가장 빠르고 YOLO는 Fast YOLO보다 mAP가 더 높게 기록되어짐.

#### VOC 2007 Error Analysis

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81650328-b3138080-946c-11ea-8c3b-c9693651c687.png" width="50%"></p>

Fast R-CNN은 localization 오류는 작지만 배경 오류 비중은 큼.

### Real-Time Detection In The Wild
Web cam에 연결하여 YOLO의 real-time 성능을 측정함.

### Conclusion
기존의 분류기 접근방식과 다르게 YOLO는 탐지 성능에 직접 대응하는 비용 함수에 바로 학습하고, 전체 모델을 공동으로 훈련함.
따라서 이미지를 단 한 번 보고, 단 한 번의 통합된 파이프라인을 거쳐서 객체를 인식하는 방법이라고 말할 수 있음.
