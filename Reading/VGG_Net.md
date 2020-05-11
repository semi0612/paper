## VGG (Very Deep Convolutional Networks For Large-Scale Image Recognition)

논문 링크 : https://arxiv.org/pdf/1409.1556.pdf

### ABSTRACT
##### 연구 수행

목적 : 통합 네트워크의 깊이가 대규모 이미지 인식 설정의 정확도에 어떤 영향을 받는지 목적으로 함.

내용 : 매우 작은 (3x3) 합성곱 필터(convolution filters)을 가진 아키텍처를 이용하여 깊이를 증가시키는 네트워크를 빈틈없이 평가한 것으로, 그 깊이를 16 ~ 19 가중치 레이어까지 밀어냄으로써 선행기술 구성에 대해 상당한 개선을 달성할 수 있음.

성과 : ImageNet challenge 2014 localization and classification track에서 각각 1위와 2위를 이룸.

#### 1. INTRODUCTION
컨볼루션 네트워크(Convolutional Networks, Convnets)이 컴퓨터 비전 분야에서 큰 비중을 차지함에 따라 여러가지 측면에서 개선사항을 찾았지만 본 논문에서는 또 다른 중요한 측면, 깊이에 대해서 다룸.

이를 위해 아키텍처의 다른 매개변수를 수정하고, 모든 계층에서 (3x3)의 컨볼루션 필터를 사용하여 네트워크의 깊이를 꾸준히 높임.

결과적으로 ILSVRC classification and localization에서 높은 정확도를 달성할 뿐만 아니라, 단순한 파이프라인(정밀 조정없이 선형 SVM으로 분류된 심층 기능)으로 우수한 성능을 얻으며, 두 가지(16layer, 19layer) 최고의 성과를 낸 모델을 출시하고 다른 이미지 인식 데이터 셋에도 적용이 가능함.

#### 2. CONVNET CONFIGURATIONS
일반 레이아웃을 설명한 다음 평가에 사용되는 특정 구성을 자세히 설명. 

##### 2-1. ARCHITECTURE

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/79068861-0c20b500-7d05-11ea-8278-7289b3d76338.png" width="50%"></p>

- Conv Layer
훈련 Convnets에 대한 입력은 고정 크기 224 x 224의 RGB 이미지이며, 사전 처리는 훈련 세트에서 계산된 평균 RGB 값을 각 픽셀에서 빼는 것이다. 이후 이미지는 conv layer 통과하며 3 x. 3필터(좌/우, 위/아래, 중앙의 개념을 포착하기 위한 가장 작은 크기)를 사용한다.  즉, 패딩은 3 x 3 레이어로 1 픽셀을 뜻한다. 구성중에서 1 x 1필터를 사용하는데, 이것은 입력 채널의 선형 변환으로 볼 수 있다. 공간 풀링으로는 5개의 max pooling layer가 수행되며 일부 Conv 레이어를 따른다. Max pooling은 stride 2와 함께 2 x 2 픽셀에서 수행을 함.

- Dense Layer
세 개의 FC(Fully- Connected) Layer 구성되며,  처음 두 개는 각각 4096개의 채널이 있고, 세 번째는 1000-way로 분류하며 마지막 레이어는 softmax를 가진다. 이 구성은 모든 네트워크에서 동일함.

공통적으로는 모든 숨겨진 층에는 activation(ReLU function)인 비선형성이 구비되어 있음. 그리고 네트워크 중 어느것도(하나를 제외하고) LRN(Local Response Normalization)을 정규화를 포함하지 않음. 정규화는 성능면에서는 향상시키지 못하면서 메모리 소비와  계산 시간을 증가시킨다.

##### 2-2. CONFIGURATIONS

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/79068868-1a6ed100-7d05-11ea-95a0-c516d3baaab8.png" width="50%"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/79068872-1e025800-7d05-11ea-8037-d71cabc4cd93.png" width="50%"></p>

(A, A-LRN, B, C, D, E)각 구성이 지남에 따라 Conv Layer 자세한 구성과 깊이을 확인할 수 있고 그에 따른 가중치 수도 증가함을 볼 수 있지만 동시에 큰 깊이에도 불구하고 가중치 수는 크지 않음도 확인이 가능함.

##### 2-3. DISCUSSION

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/79068876-235fa280-7d05-11ea-82b3-d3ce009dd7a0.png" width="50%"></p>

두 개의 3 x 3 Conv Layer가 5 x 5 Conv Layer보다 효과적인 수용필드를 가지는 것으로(두 개 사이의 공간 pooling 없이) Layer 7 x 7 size 의 효과적인 수용 필드를 가짐.

- 7 x 7 Conv Layer 대신 3개의 3 x 3 Conv Layer 사용하여 얻을 수 있는 효과.

1.  한 개의 비선형 계층 대신 세 개의 비선형 계층을 통합하여 의사결정 기능을 더욱 차별화함.	
2.  계층 3 x 3 convolutional  stack의 입력과 출력이 모두 C 채널을 가지고 있다고 가정하면, stack은 27C^2 가중치에 의해 변수화 된다. 동시에 단일 7 x 7 Conv Layer는 49C^2 매개변수를 필요로 함. 이것은 파라미터의 수를 감소할 수 있는 부분으로 연산량을 줄일 수 있는 효과를 가져옴.

1 x 1 Conv Layer의 통합은 Conv Layer의 수용 필드에 영향을 주지 않고, 의사결정 기능의 비선형성을 높이는 방법으로 여기서는 1 x 1 Conv이 본질적으로 동일한 공간에 대한 선형 투영이지만(입출력 채널의 개수는 동일), 추가적인 비선형성은 정류 함수에 도입된다. LIN(Network In Network)아키텍처에 사용되었다는 점에 유의함.

#### 3. CLASSIFICATION FRAMEWORK
Convnet 교육 및 평가 분류의 세부사항을 설명함.

##### 3-1. TRAINING
Convnet 훈련 절차는 일반적으로 Krizhevsky 외 연구진으로 구성함.
즉, 다항 로지스틱 회귀 목표를 최적화하여 훈련을 수행(위 구성 기반으로 아래와 같이 추가.)

	- Model Hyperparamiter
	Batch size - 256
	
	- Model(Conv, Dense) -기본 구성에 추가한 목록
	Dense - 처음 두개의 FC Layer 중간 Dropout(0.5) 설정
	
	- Model Compile
	Optimizer - SGD(momentum=0.9, lr(학습 속도)= first 0.01, second=0.001)

네트워크의 가중치의 초기화가 심층망에서 경사로의 불안정성으로 인해 학습을 지연시킬 수 있기 때문에 네트워크 가중치의 초기화가 중요함. 이 문제를 피하기 위해 무작위 초기화로 교육할 수 있을 정도로 얕은 구성부터 시작하여 나중에는 글로트&벤기오의 무작위 초기화 절차를 이요하여 사전 훈련 없이 체중을 초기화하는 것이 가능하다는 것을 알았음.

-Training image size

고정 크기 224 x 224 Convnet 입력 이미지를 얻기 위해 축소된 교육 이미지에서 무작위로 잘라내었음.
S(훈련 해상도)를 동위원소 축소된 교육 이미지의 가장 작은 측면으로 설정하여 Convnet 입력이 잘려나감.
S는 224 x 224로 고정되며 이상의 값을 취할 수 있음.

S(Training Scales)를 설정하기 위한 두 가지 접근법
1. 단일 스케일 훈련에 해당하는 S를 수정하는 것(S=256으로 미리 훈련된 가중치로 초기화되었으며, S=384 네트워크의 속도 향상을 위해 사용하여 실험을 함.).     
2.[Smin, Smax]의 일정 범위에서 무작위로 S를 추출하여 각 훈련 이미지를 개별적으로 축소하는 다중 스케일 훈련(광범위한 해상도에 걸쳐 물체를 인식하도록 훈련되는 스케일 지터링에 의한 훈련 세트 증대로도 볼 수 있으며 속도 상의 이유로 고정 S=384로 사전 훈련을 받은 동일한 구성을 가진 단일 스케일 모델의 모든 레이어를 미세 조정하여 멀티 스케일 모델을 교육함.)

##### 3-2. TESTING
훈련된 Convnet과 입력 이미지가 주어진다면 다음과 같은 방법으로 분류를 함.
1. 미리 정의된 가장 작은 이미지 측면으로 등방적으로 재설계되며 Q(Test Scale)로 표시. - Q와 S가 같을 필요 없음.
2. 네트워크와 비슷한 방식으로 재설계 된 Test image에 밀접하게 적용.
3. 발생한 FC Layer는 전체영상에 적용.
결과적으로 입력 이미지 크기에 따라 등급 수와 동일한 채널 수를 갖는 클래스 점수 맵과 가변 공간 분해 등이 나타남.
마지막으로 영상에 대한 클래스 점수의 고정 크기 벡터를 얻기 위해 클래스 점수 맵(class score map)은 공간적으로 평균화 됨.

전체 이미지에 Fully-Convolutional Network 적용되기 때문에, Multiple crops을 샘플링할 필요가 없으며, 각 crop에 대해 네트워크 컴퓨팅이 필요하기 때문에 효율성이 떨어짐.
Multi-Crop Evaluation은 다양한 컨볼루션 경계 조건 때문에 Dense Evaluation을 보완하며 이는 전체 네트워크 수용 분야를 상당히 증가시켜 더 많은 맥락을 포착함.

##### 3-3. IMPLEMENTATION DETAILS
단일 시스템에 설치된 여러 GPU에 대해 훈련과 평가를 수행할 수 있을 뿐만 아니라, 여러 해상도의 풀사이즈 영상에 대해서도 훈련과 평가를 수행할 수 있도록 하는 Multi-GPU 훈련은 데이터 병렬주의를 이용하며, 각 GPU에서 병렬로 처리되는 여러 GPU 배치로 각 훈련 이미지를 분할하여 수행이 가능하게 됨. 즉, 훈련 속도 측면에서 높은 효율을 가져옴.

#### 4. CLASSIFICATION EXPERMENTS
ILSVRC 2012-2014 데이터 세트에서 설명된 Convnet 아키텍처가 달성한 이미지 분류 결과 제시.
Dataset 1000 classes, training(1.3M images), validation(50K images), test(100K images, fix class label include)
분류 성과는 top-1과 top-5 두 가지로 나타내며 top-1 다중 클래스 분류 오류를 나타냄으로써 즉, 잘못 분류된 영상의 비율을 말하고 top-5는 실제 범주가 상위 5개의 예측 범주 밖에 있는 영상의 비율로 계산함.

##### 4.1 SINGLE SCALE EVALUATION
Convnet 모델의 성능을 단일 척도로 평가하는 것으로 시작으로 테스트 이미지 크기는 다음과 같이 설정을 함.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/79068886-307c9180-7d05-11ea-9fc7-66c290126835.png" width="50%"></p>

single-scale test영상을 적용했을 때의 결과는 위에 있는 표와 같다. 망이 깊어질수록 결과가 좋아지고, 학습에 scale jittering을 사용한 경우에 결과가 더 좋아진다는 것을 확인할 수 있음.

##### 4-2. MULTI-SCALE EVALUATION
Convnet 모델을 단일 해상도로 평가한 후,  Test에 스케일 지터링(scale jittering)의 영향을 평가함.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/79068889-33778200-7d05-11ea-988e-c75203e544d1.png" width="50%"></p>

S가 고정된 경우는 (S-32, S, S+32)로 Q 값을 변화 시키면서 Test를 하며, 학습의 scale과 Test의 scale이 많이 차이가 나는 경우는 오히려 결과가 더 좋지 못해 32만큼 차이가 나게 하여 Test를 진행함. 

##### 4-3. MULTI-CROP EVALUATION
Dense Convnet Evaluation를 Multi-Crop Evaluation과 비교를 하며 또한 두 평가 기법의 소프트맥스 출력을 평균화하여 상호보완성을 평가한다.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/79068894-36727280-7d05-11ea-8bd2-69bcf2c9056f.png" width="50%"></p>

학습에 스케일 지터링을 적용한 경우는 출력의 크기는 [256, 384, 512]로 Test 영상의 크기를 정했으며, 예상처럼 스케일 지터링을 적용하지 않은 것보다 훨씬 결과가 좋고, single-scale 보다는 multi-scale이 결과가 좋다는 것을 확인할 수 있음을 위에 있는 표를 통해서 확인 가능함.

##### 4-4. CONVNET FUSION
소프트맥스 클래스 사후확률(soft-max class posteriors)를 평균화하여 여러 모델의 출력을 조합하여 모델의 보완성으로 인해 성능이 향상되며 더 낮은 오류를 달성할 수 있었음.

##### 4-5 COMPARISON WITH THE STATE OF THE ART
ILSVRC-2014 분류 과제로 7개 모델의 앙상블을 사용하여 7.3%의 테스트 오차로 2위를 차지했다.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/79068897-3bcfbd00-7d05-11ea-8a14-8d9717a090a3.png" width="50%"></p>

#### 5. 결론
대량의 이미지 분류에 대한 매우 깊은 경직망(최대 19개)을 평가하며 표현 깊이가 분류 정확도에 이롭다는 것이 입증되었으며, 깊은 Convnet 아키텍처를 사용하여 ImageNet 챌린지 데이터 세트의 높은 성능을 달성할 수 있었다.






## summary
> 1. ILSVRC 2014 대회에서 GoogleNet에 근소한 차이로 아쉽게 2등을 차지한 Network
> 2. VGG Net 논문은 망의 깊이가 딥러닝 결과에 어떤 영향을 미치는 지 알아보기 위하여 연구
> 3. 3x3 filter size 사용
> 4. 이전의 depth는 8layers 수준에서 머물렀다면 GoogleNet, VGGnet 이후 크게 깊어짐
> 5. 깊은 네트워크를 가지고 있지만, GoogLeNet과 비교하면, 구조가 매우 간단하다는 장점
> 6. 매우 많은 파라미터를 이용하여 연산한다는 단점
