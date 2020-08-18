## U-Net (Convolutional Networks for Biomedical Image Segmentation)

논문 링크 : https://arxiv.org/pdf/1505.04597v1.pdf

-------------------

### Abstract
Data Augmentation을 활용함으로써 annotated sample을 보다 효율적으로 사용하는 학습 전략을 보여줌.
이 네트워크는 Contracting path, Expanding path 구조를 전반적으로 다룸.

다음은 전자 현미경(EM)이미지를 segment/annotate화 한다.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81296649-2dc74f00-90ad-11ea-9168-766f65a2edaa.png" width="50%"></p>

### Introduction & Network Architecture
Convolution Network 일반적인 용도는 이미지에 대한 출력이 클래스 레이블 분류 작업에 있었다.

그러나 많은 시간 작업이 걸리는 Biomedical Processing에서 원하는 출력은 localization을 포함해야 한다.
즉, 클래스 라벨은 각 pixel에 할당 되어야 한다는 것.

##### U-Net Network Architecture

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81297096-c3fb7500-90ad-11ea-9e2e-dce8873dbc07.png" width="75%"></p>

Contracting path은 Context를 캡쳐하며 Expanding path은 정교한 localization을 가능하게 만드는 구조를 확인할 수 있음.

##### Contracting path - Convolution Encoder
피쳐 맵 Copy and Crop하여 Concat하는 구조이다.

3 x 3 Convolution 2회 및 2 x 2 Max-pooling stride2를 연속 수행되며 downsampling시에는 2배의 featre channel을 사용함. - 고급 기능을 추출하는데 도움이 되지만 피쳐 맵의 크기는 줄어듬.

##### Expanding path - Convolution Decoder
Upsampling할 때, 조금 더 정확한 localization을 하기 위해서 위 Contracting path을 이용함.

2 x 2 Convolution(Up-convolution) 3 x 3 Conv 2회 연속 수행하여 피쳐 맵의 크기를 복구함(feature channel이 반으로 줄어듬). - "what'을 증가시키지만 "where"를 감소시킴.
즉, 고급 기능을 얻을 수 있지만, localization 정보는 잃어 버림.

Up samling 이후에는 동일한 수준의 피쳐 맵을 제공하기 위해 Contracting path에서 Expanding path로 localization 정보를 제공함.

결과적으로 출력 피쳐 맵은 2개의 클래스, 셀 및 막만을 갖기 때문에 1 x 1 Conv 전환으로 피쳐 맵의 크기를 64에서 2개의 클래스로 맵핑함.



기존에는 Sliding-window을 하면서 로컬 영역(패치)을 입력으로 제공해서 각 픽셀의 class label을 예측 했으면, 기존 방법에 2가지 단점을 보완하고자 Fully Convolution Network구조를 제안함.

2가지 단점

    - 네트워크가 각 패치에 대해 개별적으로 실행되어야 하고 패치가 겹쳐 중복성이 많기 때문에 상당히 느리다.
    - localization과 context사이에는 trade-off가 있는데, 이는 큰 사이즈의 pathces는 많은 max-pooling을 요구해서 localization의 정확도가 떨어질 수 있고, 작은 사이즈의 patches는 협소한 context만을 볼 수 있다.

##### Overlap Tile Strategy

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81299146-bdbac800-90b0-11ea-8919-33dc07db0cbf.png" width="75%"></p>

위 그림에서 전체 이미지가 부분적으로 예측됨. 자세히 보면 영상의 노락색 영역은 파란색 영역을 Patch( == Tile)단위로 잘라서 사용하여 예측함.
즉 이미지 경계에서 이미지는 미러링에 의해 얻어짐.

출력 분할 맵의 매끄러운 타일을 얻기 위해서는 입력 타일 크기를 선택하는 것이 중요함.
즉, 모든 2 x 2 Max-pooling 작업이 균등한 x와 y 사이즈의 레이어에 적용되도록 해야 함.

### Training
학습은 Stochastic gradient descent로 구현되었으며 이 논문에서는 학습 중 GPU memory의 사용량을 최대화 시키기 위해서 batch size를 크게해서 학습시키는 것 보다는 input tile의 size를 크게 주는 방법을 사용함.

따라서, 이 방법은 batch size가 작기 때문에, 이를 보완하고자 momentum의 값을 0.99를 줘서 지난 값들을 더 많이 반영하게 하여서 학습이 잘 되도록 하였음.

##### Elastic Deformation for Data Augmentation(Data Augmentation)

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81300529-bf858b00-90b2-11ea-97be-84a24d78afbd.png" width="50%"></p>

훈련 세트가 작아서 세트의 크기를 증가시키기 위해, 입력 및 출력 이미지 세그먼테이션 맵을 3 by 3 elastic 변환 행렬을 통해 수행함.
세포를 세그먼테이션 하는 것이기 때문에 성능 향상에 매우 큰 역할을 함.

##### Separation of Touching Objects

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81300763-0e332500-90b3-11ea-9d53-fa0c1bc6e26c.png" width="50%"></p>

위 그림에서 왼쪽은 분할 맵, 오른쪽은 가중치 맵을 나타냄.

서로 밀접하게 배치되어 있어 네트워크에 의해 쉽게 병합되고 분리되어 가중치 맵이 출력에 적용됨.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81301065-8568b900-90b3-11ea-84aa-04587fdf1a26.png" width="50%"></p>

위의 식은 가중치 맵을 계산하기 위한 식이 d1(x)는 위치 x에서 가장 가까운 셀 경계까지의 거리이고, d2(x)는 두번째로 가까운 셀 경계까지의 거리이다. 따라서 border에서는 수치에서와 같이 무게가 높음.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81301098-9285a800-90b3-11ea-91e2-571b4fdce746.png" width="50%"></p>

따라서 교차 엔트로피 기능은 가중치 맵에 의해 각 위치에서 불이익을 받음.
또한 네트워크가 Touch Cell 사이의 작은 분리 경계를 학습하도록 하는데 도움이 됨.

### Experiments

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81302515-7256e880-90b5-11ea-9108-03d437f5dae6.png" width="50%"></p>

ISBI 2012 Challenge에서 Rank 1위를 거둠.
Warping Error, Rand Error, Pixel Error, 훈련 시간, 테스트 속도에서 좋은 성능을 얻음.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81302544-7a168d00-90b5-11ea-8983-795f8b859e41.png" width="50%"></p>

PhC-U373, Dlc-HeLa 데이터 세트에서 가장 높은 IoU를 얻음.

### Conclusion
U-Net 구조는 Elastic 변환을 적용한 Data Augmentation 덕분에 매우 다른 biomedical segmentation applications에서 좋은 성능을 보였다. 그리고 annotated image가 별로 없는 상황에서 매우 합리적임.

U-Net 구현은 Caffe 기반으로 제공되며 다양한 task에서 쉽게 적용되어 사용될 것이다.
