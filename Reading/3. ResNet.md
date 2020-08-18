## ResNet (Deep Residual Learning for Image Recognition)

논문 링크 : https://arxiv.org/pdf/1512.03385.pdf

### summary
```
* shortcut connection을 사용하면서 연산량은 증가했지만 학습 파라미터 수에는 크게 영향을 주지 않았다.
* 더 많은 수의 레이어를 가지는 깊은 모델도 잘 학습되었다.
* 학습의 초기 단계에서 Residual net의 수렴 속도가 plain network보다 빠른것을 확인할 수 있었다.
=> 네트워크가 깊어질수록 발생하는 문제점을(problem of vanishing/exploding gradients) 해결하고자 Residual 네트워크을 이용하였다.
```
-------------------

### ABSTRACT
이전에 사용했던 것보다 상당히 깊은 네트워크의 훈련을 쉽게 하기 위해 Residual Learning Framework를 제시. 이는 계층을 참조되지 않은 함수를 학습하는 대신 계층 입력에 대한 학습 잔량 함수로 명시적으로 재조정. 

Residual Networks가 최적화하기 쉽고 상당히 증가된 깊이에서 정확성을 얻을 수 있음을 보여주는 종합적인 결과를 제공.

VGGNet보다 8배 깊은 최대 152 Layer를 가지며 이는 시각적 인식 작업에서 중요한 역할을 가짐.

Detection, Localization, Segmentation, Classification 부분에서도 좋은 결과를 얻었으며 수상 부분은 생략함.

#### 1. INTRODUCTION
깊은 Convolutional Neural Networks는 이미지 분류를 위한 일련의 획기적인 발전을 가져옴. 또한 시각적 인식 작업도 매우 깊은 모델로부터 큰 이점을 얻음.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75325679-92e81280-58bc-11ea-82b3-2720572286d2.png" width="50%"></p>

- 망이 깊어질수록 나타나는 문제점:

다음 위 그래프를 보면 네트워크가 깊을수록 학습 오류가 높아 테스트 오류가 발생하는 것을 확인할 수 있음. 그래서 깊이의 중요성에 의해 다음과 같은 의문(많은 레이어를 쌓을수록 더 나은 네트워크를 학습하는 것이 쉬운가?)에 대해서 해결하고자 함.
이 의문은 Vanishing/Exploding(소멸/폭발) 문제로 처음부터 수렴을 방해함. 즉, 네트워크 깊이가 증가하면 정확도가 포화되고, 그후 빠르게 감소한다. 단, 오버피팅은 아니다.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75325687-95e30300-58bc-11ea-8f6f-8553b1cd1625.png" width="50%"></p>

본 논문에서는 deep residual learning framework을 도입하여 열화 문제를 다룬다.
Residual mapping(잔차 매핑)을 최적화하는 것이 더 쉽다는 가설 세움. 위 그림에서 F(x) + x 공식은 (shortcut connections이 있는 feedforward neural networks)을 통해 실현할 수 있다. 여기서 shortcut connections은 하나 이상의 레이어를 건너뛰는 것을 말함. 
그렇게 얻을 수 있는 성능으로는 Deep residual nets 깊이가 크게 증가하여 정확도가 쉽게 향상되어 이전 네트워크보다 상당히 우수한 결과를 얻을 수 있음.

#### 2. Relaated Work
Residual Representations은 이미지 인식에서 VLAD는 사전과 관련하여 잔차 벡터에 의해 인코딩되는 표현이며, Fisher Vector는 VLAD의 확률 버전으로 제작 할 수 있음. 둘 다 이미지 검색과 분류를 위한 강력하고 얕은 표현함. 잔차 벡터를 인코딩하는 것이 훨씬 빠르게 수렴되는 것으로 나타남.

Shortcut Connections. Vanishing/Exploding gradients을 해결하기 위해 몇개의 중간 레이어가 보조 분류기에 직접 연결됨. 
레이어 응답, 경사 및 전파된 오류를 중앙에 맞추기 위한 방법으로 '시작'레이어는 shortcut 분기와 몇 개의 더 깊은 분기로 구성됨. "highway networks"는 게이팅 기능과 함께 shortcut connections을 제공하며 shortcut가 '닫힘'에 있으면 highway entworks의 레이어는 비잔차 기능을 나타내고 반대로 identity shortcuts은 절대 닫히지 않으며 학습할 추가 잔차 기능과 함께 정보가 항상 전달됨.

#### 3. Deep Residual Learning

##### 3-1. Residual Learning

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75325699-99768a00-58bc-11ea-912b-64bf972cfc05.png" width="50%"></p>

위 그림에서 H(x)를 얻는 것이 목표가 아니라 H(x) - x를 얻는 것으로 목표를 수정한다면 출력과 입력의 차를 얻을 수 있도록 학습을 하게 되면 2개의 weighted layer는 H(x) - x를 얻도록 학습이 되어야 한다. 따라서 F(x) = H(x) - x라면 결과적으로 H(x)는 H(x)=F(x) + x가 됨. 

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75325706-9bd8e400-58bc-11ea-8602-1795d5216255.png" width="50%"></p>

따라서 다음과 같은 구조를 가질 수 있음. 이것은 Residual Learning의 기본 구성이며 입력에서 바로 출력으로 연결되는 shortcut 연결이 생기게 되었으며, 이 shortcut는 파라미터가 없이 바로 연결외 되는 구조로 연산의 크게 영향을 끼치지 않음.
결과적으로 identity 매핑으로 깊은 망에 대해서 최적화, 정확도 부분에서 좋은 성능을 이끌어 낼 수 있었음.

##### 3-2. Identity Mapping by Shortcuts

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75325712-9f6c6b00-58bc-11ea-822c-89fc2ca13ff5.png" width="50%"></p>

여기서 x와 y는 고려된 레이어의 입력 및 출력 벡터를 나타내며 함수 F(x, {Wi})는 학습할 잔차 매핑을 말함. 위의 식은 shortcut connection은 추가 매개변수나 계산 복장섭을 도입하지 않음. 이것은 동일한 개수의 매개 변수, 깊이, 너비 및 계산 비용을 동시에 갖는 일반 / 잔차 네트워크를  공정하게 비교 가능하게 함.
반면에, 아래 식에는 입력/출력 채널을 변경할 때 shortcut connection로 선형 투영 Ws는 치수를 일치시킬 때만 사용됨을 보여줌.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75325725-a2675b80-58bc-11ea-9a07-9005a22b7e51.png" width="50%"></p>

위에 식들은 표기법 단순화를 위해 완전히 연결된 레이어에 관한 것이지만 컨볼루션 레이어에도 적용할 수 있음.

##### 3-3. Network Architectures
인스턴스를 제공하기 위해 ImageNet에 대한 두 가지 모델을 다음과 같이 설명을 함.

- Plain Network 
다음 아래 그림에서는 VGGNet에서 영감을 얻었음을 보여줌.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75325731-a5fae280-58bc-11ea-9e93-6b9196081050.png" width="40%" height="50%"></p>

그림 왼쪽 VGG-19, 중간 34개의 매개변수 레이어 평이한 네트워크, 오른쪽 34개의 매개변수 레이어 잔차 네트워크를 보여줌. 
총 가중 계층 수는 34개이며 VGGNet보다 필터가 적고 복잡성이 낮다는 것을 보여줌으로 34 계층 기준에는 36억 개의 FLOP(곱하기)가 있으며 이는 VGG-19(1896억 FLOP)의 18%에 불과함.

- Residual Network
위 그림의 잔차 함수는 identity 매핑은 입력과 출력이 동일한 치수일 때 직접 사용할 수 있으며 치수가 증가할 때는 두가지 옵션을 고려함.

1. 지름길은 여전히 identity mapping을 수행하며, 치수를 증가시키기 위해 추가적인 제로 패딩을 함. 추가 매개변수를 도입하지는 않음.
2. 예상 지름길은 치수와 일치하기 위해 사용함.

두 경우 지름길이 두 크기의 피쳐 맵에 걸쳐 있을 때 2의 보폭으로 수행됨.

##### 3-4. Implementation
ImageNet을 기반으로 이미지 크기는 스케일 확대를 위해 [256, 480]에서 무작위로 샘플링한 짧은 면으로 조정함. 224x224 크롭은 픽셀당 평균을 밴 채 이미지 또는 어떤 수평 플립으로 무작위로 샘플링되며 표준 색상 증대를 사용함.

- Training
추가적인 model 구성
  
  - Model Hyperparamiter
  Batch_size 256
  - Model 추가적인 구성
  Conv_2D와 activation 사이의 BN(배치 정규화) 추가.
  Dropout을 사용하지는 않음.
  - Model compile
  Optimizer SGD(lr = 0.0001 momentum = 0.9) 
	
- Testing
10-crop 테스트 채택하며 최상의 결과를 위해 fully-convolutional을 채택하며 복수의 해상도 점수를 평균화 함(이미지는 더 짧은 쪽이 {224, 256, 384, 480, 640}에 있도록 크기가 조정).

#### 4. Experimnets

##### 4-1. ImageNet Classification
1000개의 클래스로 구성된 ImageNet 2012 Classification  평가.
Dataset으로는 training 1.28million images, validation 50k images, test 100k images 사용.
Top-1 and top-5 error 평가함.

- Plain Networks.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75325739-ab582d00-58bc-11ea-9684-4b0004ce5d94.png" width="50%"></p>

우선 18, 34layer Plain Networks를 평가함.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75325747-b14e0e00-58bc-11ea-9a37-693a4e122c63.png" width="50%"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75325752-b4e19500-58bc-11ea-82f7-1d2c307dc756.png" width="50%"></p>

다음은 Top-1 error 보여줌으로써, 34 layer가 얕은 18 layer 보다 plain network에서 더 높은 vlidation error 확인할 수 있음. 그래서 앞으로 나오는 관찰은 train/validation 검사 오류를 비교함.
이 최적화 어려움이 경사 사라짐으로 인해 발생할 것 같지는 않다고 주장하며 Plain Networks는 BN으로 교육되어 전방의 전파 신호가 0이 아닌 분산을 갖도록 보장함. 그래서 앞 또는 뒤의 신호도 소멸하지 않음.

이후 많은 훈련 반복 실험을 했고 여전히 성능이 저하되는 문제를 관찰했으며, 단순히 더 많은 반복을 사용한다고 해결할 수 없다는 것을 알았음.

- Residual Networks
18, 34layer Residual Network를 평가함.
기본 구조는 동일하며 3x3 필터에 지름길 연결을 추가하며 identity매핑을 사용하며 차원을 증가시키기 위해 제로패딩을 사용한다. 그래서 일반적인 것에 비해 특별한 매개변수가 없음을 확인할 수 있음.
결과적으로 위에 있는 오른쪽 그래프를 보면 34-layer에서 더 낮은 error 보여주면서 Train은 확연히 낮은 것을 보여줌으로써 아래 표를 보면 검증 데이터에 일반화할 수 있었음.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75325761-b8751c00-58bc-11ea-99c2-f9a4ce54d00f.png" width="50%"></p>

- Identity vs Projection Shortcuts 
매개변수가 없는 identity 지름길이 훈련에 도움이 된다는 것을 보여줌.
위 왼쪽 그림에서 세가지(A, B, C) 지름길 예상도를 비교함.
A) 제로 패딩 지름길은 치수를 증가시키는데 사용. 모든 지름길 매개변수는 없음.
B) 치수 증가를 위해 사용. 
C) 모든 지름길을 예상. 

간단하게  말하고자 하는 것은 세가지 지름길 예상도 모두 Plain Network보다 상당히 낮음을 보여줌.

- Deeper Bottleneck Architectures.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75325770-bca13980-58bc-11ea-9497-d4ba1d4ed206.png" width="50%"></p>

위 왼쪽 그림은 34-layer에서 블록이며 오른쪽 그림은 50, 101, 152-layer "bottleneck" buliding block 나타냄.

더 깊은 네트워크에 대해 그리고 Training 시간에 대한 우려 때문에  "bottleneck" buliding block 같은 설계로 수정함.
자세히 관찰하면 2에서 3 스택으로 변경하였으며 1 x 1, 3 x 3, 1 x 1컨볼루션으로 치수를 줄인 후 증가하는 역할을 하며 3 x 3 layer 더 작은 입출력 치수를 가진 병목현상이 됨.
결과적으로 시간 복잡성과 모델 크기면에서 효율적인 모델이 될 수 있음.

- 50, 101, 152-layer ResNets
VggNets보다 복잡성이 더 낮으며 34-layer 보다 상당한 차이로 더 정확성을 가짐을 위에 있는 연속적인 표를 통해서 확인 가능함.

- Comparisons with State-of-the-art Methods
Top-1, top-5 검증 오류로 더 나은 결과를 가져왔으며 ILSVRC 2015에서 1위를 차지했음.

-Exploding Over 1000 lyaers.
1000개가 넘는 layer로 구성된 매우 깊은 모델을 탐구함.
1202-layer network  테스트 오차는 여전히 꾀나 좋았지만 시험 결과는 110-layer network의 시험결과보다 더 나쁨.  이것을  overfitting으로 판단하며 작은 dataset에 불필요하게 깊은 네트워크를 구성했다고 판단함.
따라서 최상의 결과를 얻기 위해 maxout or dropout같은 정규화를 적용함.

이후 detection PASCAL, MS COCO, ImageNet Localization ImageNet 부분은 생략함.
