
## FPN (Feature Pyramid Networks for Object Detection)

논문 링크 : https://arxiv.org/pdf/1612.03144.pdf

-------------------

### Abstract
Feature pyramids는 recognition의 기본 요소이다. 이는 서로 다른 Scale 개체를 탐지하는 시스템에서 compute, memory 면에서 pyramid 표현에 어려움이 있다.

이 논문에서는 다중 Scale을 이용하며 약간의 추가 비용으로 Feature pyramids를 이용하기 위한 depth convolutional networks의 pyramid 계층 구조를 지닐 수 있으며 이것은 측면 연결부가 있는 topdown architecture에 의해 개발된다.

이 기법은 일반적인 feature extractor로서 상당한 개선올 보여줬으며 Basic Faster R-CNN System에서 FPN을 사용하여, COCO2016 challenge에서 우수한 성적을 얻음.

### Introduction

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/85272681-7600cc00-b4b7-11ea-8983-df0f2d419947.png" width="75%"></p>

object detection 분야에서 scale-invariant는 scale과 location에 상관없이 객체를 하나의 class로 탐지하기 위한 아주 중요한 문제이다.

이전에는 다양한 크기의 물체를 탐지하기 위해 이미지 자체 크기를 resize(pooling or stride convolution)하여 detection layer로 객체를 탐지하였음. - SSD, RFB(1-stage method)

이를 개선하기 위해 본 논문에서 소개하는 Feature Pyramid Network(FPN) 방법을 소개하고자 함.

#### Featureized Image Pyramid

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/85274087-71d5ae00-b4b9-11ea-98a0-8881bffd3e91.png" width="50%"></p>

각 feature map에서 독립적으로 특징을 추출해 객체를 탐지하는 방법으로 연산량과 시간 관점에서 비효율적이며 parctical하게 적용하기 어려움.

#### Single Feature Map

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/85274321-bfeab180-b4b9-11ea-98c5-04c406087ee9.png" width="50%"></p>

convolution layer가 scale variant에 robust한 점을 이용하는 방법으로 convolution layer를 통해 feature를 압축하는 방식이다. 하지만 끝에 압축된 feature만을 사용함으로 성능이 떨어짐. 이는 YOLO V1에서 사용하였던 방법이었음.

#### Pyramidal Feature Hierarchy

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/85274607-21128500-b4ba-11ea-888b-dd4e0dbe56dd.png" width="50%"></p>

서로 다른 scale의 feature map을 이용해 multi scale feature를 추출하는 방식으로 각 단계에서 독립적으로 feature를 추출하여 객체를 탐지함으로써 재사용하지 않는 다는 특성을 가지고 있음. 이는 SSD에서 사용하였던 방법이었음.

#### Feature Pyramid Network(FPN)

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/85360653-9cc11000-b554-11ea-8b57-fd62eff0c16f.png" width="50%"></p>

top-down방식으로 feature를 추출함. - 각 추출된 feature map의 low-resolution 및 high-resolution feature들을 묶는 방식

각 계층에서 독립적으로 feature를 추출해 객체를 탐지함. - 이 과정에서 상위 계층에서 이미 생성된 feature map을 재사용하므로 multi-scale feature들을 효율적으로 재사용 가능함

위의 전반적인 과정은 다음과 같음.

    - 임의 크기의 단일 scale 영상을 입력 받는 CNN은 각 layer를 거치면서 pyramid 구조를 만들며 더 많은 semantic 정보를 갖는 다양한 feature를 얻게 됨. 
        * 각 계층 prediction 과정을 넣어서 scale 변화에 더 강인한 모델로 구성
    - FPN은 skip connection, top-down, CNN forward pass 과정에서 생성되는 pyramid의 구조를 가짐. 
        * forward pass에서 생성된 top-down 과정 upsampling하여 spatial resolution을 올리며 손실된 localization feature들을 skip-connection을 이용하여 보충하여 scale variant에 강인한 feature map을 생성 할 수 있음
    - 따라서 FPN은 Convolution Network의 피라미드 특징 계층 구조인 low ~ high level의 semantic feature들을 모두 갖고 있으며 전반적으로 높은 수준의 semantic information을 포함하는 feature pyramid를 생성하게 됨.
    - 이 FPN은 Faster R-CNN과 Faster R-CNN의 Region Proposal Network(RPN)을 기반으로 함.
  
  feature concat process는 backbone network와 독립적으로 진행됨.
  ###### 논문에서는 ResNet을 backbone으로 사용하여 예시를 둠.

  이어서 위 그림에서 크게 두 부분으로 구성하는 왼쪽 부분인 Bottom-up pathway, 오른쪽 부분인 Top-down pathway and lateral connections(skip-connection)에 대해 나눠서 설명하고자 함.

##### Bottom-up pathway
backbone network(network에 따라 상대적)의 feed forward 계산과정의 forward 단계에서 매 레이어마다 semantic을 응축하는 역할을 함.

깊은 모델일수록 가로, 세로 크기가 같은 레이어들이 여러 개 있을 수 있다. 그래서 여기서 같은 크기를 갖는 레이어들을 하나의 단계로 취급해서 각 단계의 맨 마지막 레이어를 skip-connection으로 연결함.

각 단계 마지막 레이어의 출력은 feature map의 reference set으로 선택함.

피라미드의 feature이 다양하기 위해서는 각 단계의 가장 깊은 레이어에는 가장 의미 있는 feature가 있어야 하며 이것은 간 단계 마지막 feature map을 이용해 skip-connection 형성함.

##### Top-down pathway and lateral connections(skip-connection)
하향식 과정에서는 많은 semantic 정보들을 갖고 있는 feature map을 2배로 upsampling해 더 높은 해상도의 이미지를 만들며 여기서 skip-connection을 통해 같은 사이즈의 bottom-up layer와 합쳐서 손실된 local 정보를 보충해줌.


<p align="center"><img src="https://user-images.githubusercontent.com/45933225/85362120-a0569600-b558-11ea-88aa-f40b44273f8c.png" width="50%"></p>

매 레이어마다 classifier와 regressor 적용되며 여기서 같은 하나의 classifier와 regressor를 사용하기에 모두 같은 256 채널의 feature map을 입력 받아야 함. 따라서 skip-connection을 할 때 1 x 1 convolution(pointwise)으로 채널을 맞춰주게 됨.

그리고 위 그림을 보면 알 수 있듯이 이 프로세스는 마지막 resolution의 feature map이 생성될때까지 동일하게 수행됨.

마지막으로, 합쳐진 각 feature map에 3 x 3 convolution을 수행해 upsampling의 aliasing 효과를 줄인 후 최종 feature map을 생성함.

따라서 이렇게 생성된 feature map은 각 prediction 과정을 거치며 (P 2, 3, 4, ...)이 생성되며, 동일한 spatial size인 (C 2, 3, 4, ...)에 대응되어 lateral connection을 형성한다.

###### Lateral connection의 필요성
top-down layer도 높은 semantic을 갖고 있지만, 지역적인 정보가 부족하게 되어 정확도가 떨어지게되는 단점이 생길 수 있다.

###### Pyramid 구조의 필요성
크기가 다른 각 layer에서 예측하는 것이 아니라 top-down의 맨 마지막 레이어에서만 prediction하면 baseline보다는 높지만 위에서 소개한 구조보다는 성능이 떨어짐. 왜냐하면 scale variant에 덜 robust해지기 때문이다.

또 피라미드 구조의 맨 마지막의 해상도가 높기 때문에 많은 anchor들이 적용되었는데도 그다지 성능이 좋지 않음. 이는 anchor가 많은 것이 성능을 높히기 충분하지 않다는 것을 반증한다.

### Application
위에서도 언급했지만 이 FPN은 Faster R-CNN과 Faster R-CNN의 Region Proposal Network(RPN)을 기반으로 한다. 그래서 다음 아래와 같이 두 부분으로 나눠서 자세히 설명을 한다.
  
#### Feature Pyramid Networks for RPN

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/85364624-7ef8a880-b55e-11ea-9f28-7c590e27495d.png" width="50%"></p>

RPN을 위해 3 x 3 convolution, sibling 1 x 1 convolution을 각 계층에 추가 구성 하며 각 feature 마다 bounding box regression, label classification 수행함.

각각의 pk에 single scale anchor box를 사용하며 feature map 크기에 따라 비례하여 적용시킴. 이는 multi-scale detection을 위함.

IoU threshold는 0.7 이상은 positive, 0.3 이하는 negative Sharing parameter는 Sibling 1 x 1 convolution의 parameter를 모든 FPN이 공유하도록 한다.

#### Feature Pyramid Networks for Fast R-CNN

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/85364322-db0efd00-b55d-11ea-8421-917514fdbe46.png" width="50%"></p>

FPN을 통해 다양한 scale의 feature map을 생성했으며 가장 해상도가 높은 feature map을 RPN에 적용시킴.

각 region proposal을 적당한 크기의 feature map pk에 적용시킴.
    <p align="center"><img src="https://user-images.githubusercontent.com/45933225/85364143-76ec3900-b55d-11ea-8045-a93e391afd35.png" width="50%"></p>

그 다음 Rol pooling와 FC layer 단계를 거쳐 bounding box regression and label classification의 결과를 반환 받음.

### Experiments on Object Detection
MS COCO 데이터와 ResNet 모델 기반으로 실험.

Region Proposal with RPN으로는 Synchronized SGD, mini-batch size 2을 사용함.

#### Ablation Experiments

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/85365395-3f32c080-b560-11ea-8b57-db234889bb3f.png" width="50%"></p>

RPN에서 single layer를 하나 더 사용하는 것은 Coarser resolutions과 stronger semantics의 trade-off 때문에 큰 효과가 없음. 따라서 다양한 scale의 feature map을 만드는 FPN의 효율적인 성능을 확인할 수 있다.

Ablation experiments follow에서는 위에서 언급했던 각 부분적으로 나눠서 각 효율적인 성능에 대해서 확인 할 수 있으며 다 합쳐졌을 때 low~high level의 semantic 성능이 효율적으로 발휘되는 것을 볼 수 있다.

#### Extensions: Segmentation Proposals

Segmentation prosal을 만들기 위해 FPN을 사용하며 그에 따른 결과를 확인할 수 있음.
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/85366209-d9dfcf00-b561-11ea-8cb3-044785cc6bc3.png" width="50%"></p>

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/85366135-b3219880-b561-11ea-9e51-b408b3507449.png" width="50%"></p>

###### 자세한 부분은 생략함.

### Conclusion
다양한 scale의 object를 찾기 위한 방법이며 Robustness와 performance 모두 향상되는 것을 알 수 있었다. 그리고 Bottom-up pathway, Top-down pathway, lateral connection 모두 중요한 단계임을 실험에서 보여주며, Image pyramid에 대해 일일히 계산하지 않아도 되기에 Fast R-CNN에 적용했을 경우도 더 빠르고 정확한 결과를 얻을 수 있었다.
