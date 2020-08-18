## Squeeze-and-Excitation Networks

논문 링크 : https://arxiv.org/pdf/1709.01507.pdf

-------------------

### Abstract
소개 및 목표 : Convolution은 각 계층의 로컬 수용 영역 내에서 공간적 정보와 채널적 정보를 모두 융합하여 네트워크가 정보적 기능을 구성할 수 있음. 공간적 요사를 조사하여 인코딩의 품질을 향상시킴으로써 CNN의 표현력을 강화하고자 함.

연구 : 채널 관계에 초점을 맞추고, 채널 간 상호의존성을 명시적으로 모델링하여 채널-지속적인 형상 반응을 적응적으로 재조정하는 'SE Block'이라고 하는 새로운 구조 단위를 제안.

달성 : 'SE Block'들을 쌓음으로써 서로 다른 데이터셋에 걸쳐 매우 효과적으로 일반화하는 SENet 아키텍처를 형성할 수 있었음. 다만, 약간의 추가 계산 비용이 들었음.

수상 : ILSVRC 2017 classification 1위를 차지 했다.

###### Index Terms - Squeeze-and-Excitation, Image representations, Attention, Convolutional Neural Networks.

#### 1 INTRODUCTION
CNN(Convolutional neural network)이 발전함에 따라 컴퓨터 비전에서 이미지의 특성만을 포착하여 성능을 향상시키는 보다 강력한 표현을 찾는 것으로 이 모델은 널리 사용됨.
비전 과제에 널리 사용되는 모델로써 특징들 간의 공간적 상관관계를 포착하는데 도움이 되는 네트워크로 학습 메커니즘을 통합함으로써 강화될 수 있으면서 inception 아키텍처에 의해 성능 향상을 달성하기 위해 네트워크 모듈로 다중 규모의 프로세스를 통합함. 추가적으로 공간 의존성을 더욱 더 모델링하고, 네트워크 구조에 공간 주의를 통합하는 것을 추구함.

본 논문에서는 네트워크 설계의 다른 측면, 채널 간의 관계를 조사함.
그래서 추상적 특징의 채널 간 상호의존성을 명시적으로 모델링함으로써 네트워크가 표현하고자 하는 것을 향상시키는 것을 목표로 함.
이를 위해 네트워크가 기능 재교정을 하는 방법으로 선택적으로 강조하고 덜 유용한 기능을 억제하기 위해 글로벌 정보를 사용하는 방법을 배울 수 있었음.
 
 <p align="center"><img src="https://user-images.githubusercontent.com/45933225/75980001-04f7d180-5f25-11ea-91f8-bd47a3fa3584.png" width="80%" height="50%"></p>
 
위 그림은 Squeeze-and-Excitation Block 구조를 보여줌.

간단하게 말하자면, Squeeze and excitation (SE) 블록은 모든 채널을 동일하게 취급하는 대신 각 채널에 가중치를 부여하는 방법이다.
 
형상(H, W, C) 재교정을 수행하기 위한 해당 SE Block을 구성할 수 있으며  형상 U는 공간적 차원(H x W)에 걸쳐 피쳐 맵을 집계하여 채널을 설명하는 squeeze 작업을 먼저 통과함. 이 부분은 channel-wise feature 응답으로는 전역에 걸쳐 내포화를 제작하여 네트워크의 모든 계층에 의해 수용 영역의 정보를 사용할 수 있도록 해주는 것이다.
피쳐 맵을 집계는 excitation 작업으로는 임베딩을 입력으로 하고 채널당 변조 가중치를 수집하는 단순한 자기 게이트 메커니즘의 형태를 취함. 이한 가중치는 U에 적용되어 네트워크의 후속 계층에 직접 공급될 수 있는 SE Block의 출력을 생성함.
이러한 블록을 쌓으면 SENet을 구축할 수 있다. 더욱 이러한 SE Block은 네트워크 아키텍처 깊이 범위에서 원래 블록의 drop-in 대체로도 사용될 수 있음.
깊이에서 수행하는 역할로는 이전 계층에서는 계급에 구애 받지 않는 방식으로 정보 기능을 자극하여 공유된 low-level의 표현을 강화한다면 이후 계층에서는 SE Block이 점점 전문화되어 classification가 높은 방식으로 서로 다른 입력에 대응한다.

결과적으로 SE Block에 의해 수행되는 형상 재교정의 이점은 네트워크를 통해 축적될 수 있었으면서 구조가 간단하고 성능이 효과적으로 향상될 수 있고 SE components로 교체하여 기존의 최첨단 아키텍처에서 직접 사용할 수 있음. 더불어 계산적으로 가벼우며 모델 복잡성과 계산 부담의 약간만 증가시키며 특정 dataset이나 작업에만 국한되지 않는다.

#### 2 RELATED WORK

깊이를 증가시키면서 네트워크가 학습할 수 있는 표현이 향상되며 BN(Batch Normalization)으로 안정성을 추가하고 보다 부드러운 최적화 표면을 생성하였으며 Residual기능을 통해 깊고 강력한 네트워크를 학습 할 수 있었음.
이러한 작업에 이어, 심층 네트워크의 학습 및 표현 특성에 대한 개선을 보여주는 네트워크 계층간의 연결에 대한 추가적인 부분도 있었음.
대안적이지만 네트워크 내에 포함된 계산 요소의 기능적 형태를 개선하는 방법에 초점을 맞추었으며 이전 작업에서 교차 채널 상관관계는 공간 구조에 독립적으로 또는 1 x 1 kernel 과 함께 표준 컨볼루션 필터를 사용하여 공동으로 새로운 형상으로 매핑하였다. 이 연구의  대부분은 채널 관계가 지역 수용 분야와 함께 인스턴스 제한 기능 구성으로 모델 및 계산 복잡성을 줄이는 목표에 집중되어 왔다. 
이와 대조적으로 유닛에게 글로벌 정보를 사용하여 채널들 사이의 동적 비선형 의존성을 명시적으로 모델링하는 메커니즘을 제공하면 학습 과정을 용이하게 하며 네트워크의 표현력을 크게 향상시킬 수 있었음.

Algorithmic Architecture Search.
네트워크 토폴로지를 검색하는 방법 확립과 라마르크 상속과 차별화된 아키텍처 검색을 기반으로 하이퍼 파라미터 최적화를 공식화함으로써 무작위 검색 및 정교한 모델 기반 최적화 기법을 사용하여 여러 문제를 해결할 수도 있었다. SE Block은 이러한 검색 알고리즘의 원자 구성 블록으로 사용될 수 있으며 동시 작업에서 이 용량에 매우 효과적인 것으로 나타남.

Attention and gating mechanisms.
Attention는 사용 가능한 계산 자원의 배분을 신호의 가장 유용한 구성요소로 편중하는 수단으로 해석할 수 있다. Attention 메커니즘 시퀀스 학습, 영상 등 효용성을 입증하며 이러한 애플리케이션에서 양형 간 적응을 위한 상위 수준의 추상화, 공간과 채널 관심의 결합 기존 메커니즘과 대조적으로 SE Block은 채널-시선 관계를 계산적으로 효율적인 방법으로 모델링함으로써 네트워크의 표현력을 향상시키는데 초점을 맞춘 경량 게이트 메커니즘으로 구성됨.

#### 3 SQUEEZE-AND-EXCITATION BLOCKS

Squeeze-Excitation 블록은 변환 Ftr을 기반으로 구축될 수 있는 계산 단위로서, 맵을 특징으로 하기 위해 X∈R ^H'xW'xC' 매핑함.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75980010-0c1edf80-5f25-11ea-8350-1ea452410912.png" width="50%"></p>
 
X의 해당 채널에서 작용하는 Vc의 단일 채널을 나타내는 spatial kernel이다.
Convolutional에 의해 모델링된 채널 관계는 본질적으로 implicit and local 이다.(최상위 계층 제외.) 그래서 네트워크가 후속 변환에 의해 이용될 수 있는 정보 기능에 대한 민감도를 증가시킬 수 있도록 채널 상호의존성을 명시적으로 모델링함으로써 결합 형상에 대한 학습이 향상될 것으로 기대하며 다음 변환에 투입되기 전에 글로벌 정보에 대한 접근을 제공하고 피터 응답을 두 단계로 다시 보정함.

##### 3.1 Squeeze: Global Information Embedding

각 채널에 대한 신호를 고려하여 각각의 학습된 필터는 로컬 수용 필드를 사용하며 변환 출력 U는 지역 외부의 상황 정보를 이용할 수 없다.
그래서 채널에 글로벌 공간 정보에 대한 글로벌 평균 풀링을 넣는 것을 제안하며, 채널별 통계를 생성이 가능하다.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75980023-10e39380-5f25-11ea-8d90-ca574e7cbeed.png" width="50%"></p>
 
위에 식은 통계 z ∈ Rc는 공간 치수 H x W를 통해 U를 수축시킴으로써 생성됨.  전체 이미지에 대해 표현되는 지역적 이미지의 집합으로 해석할 수 있음.

##### 3.2 Excitation: Adaptive Recalibration

Squeeze으로 얻은 정보를 활용하기 위해 채널의 의존성을 완전히 포착하는 것을 목적으로 함. 이 목적을 달성하기 위해서는 함수가 두 가지 기준을 충족해야 함.
	
  	- 유연해야 함.(채널 간 비선형 상호작용을 학습할 수 있어야 함.)
	- 복수 채널이 강조될 수 있도록 하기 위해 비수행적 관계를 배워야 함.(단열 작동)

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75980032-14771a80-5f25-11ea-837b-96a344a0a508.png" width="50%"></p>
 
위와 같은 기준을 충족하기 위해서 sigmoid 활성화가 있는 메커니즘을 사용한다. 여기서 δ = ReLU function, W ∈ R 가리킴. 
모델의 복잡성을 제한하고 일반화를 돕기 위해 비선형성 주위에 완전히 연결된 레이어 두 개로 병목 현상을 형성하여 게이트 메커니즘을 매개변수로 함.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75980039-18a33800-5f25-11ea-990b-0f4207f01846.png" width="50%"></p>

블록의 최종 출력은 활성화로 U를 다시 정렬하여 얻음.
F(u, s)는 스칼라 Sc와 피쳐 맵 Uc ∈R^H*W 채널 - 곱셈을 가리키며 Excitation은 입력 특정 설명자 z를 채널 가중치 집합에 매핑함.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75980057-1e991900-5f25-11ea-9886-18795dc7aedc.png" width="50%"></p>

결과적으로 위 그림에서 SE Block 역할로 지역 수용 영역에 국한되지 않는 채널에서 자기 주의 기능으로 간주될 수 있다.

##### 3.3 Instantiations

SE Block의 유연성은 표준 컨볼루션 이상의 변환에 직접 적용할 수 있다. 이점을 설명하기 위해 조금 더 복잡한 아키텍처의 몇 가지 예로 통합하여 SeNets을 개발함.
Inception, residual의 실험을 소개로 ResNext, Inception-ResNet, MobileNet 및 ShuffleNet와 통합하는 추가 변형 모델 SENet을 구성할 수 있었음.

결과적으로 SE Block의 유연성은 아키텍처에 통합될 수 있는 몇 가지 실행 가능한 방법이 있다는 것으로 통합 전략에 대한 민감도를 평가하기 위해 다양한 설계를 가능하게 함.

#### 4 MODEL AND COMPUTATIONAL COMPLEXITY

SE Block을 설계함으로 성능 향상과 모델 복잡성 증가 간에 좋은 절충을 제공해야 함.
다음 아래 그림과 같이 ResNet-50, SE-ResNet-50의 비교로 확인이 가능함.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75980070-235dcd00-5f25-11ea-870b-85ec63707831.png" width="50%"></p>

Squeeze(Global average pooling), excitation(two small FC layer)사용하며 가벼운 채널 측면 스케일링 작동을 사용함. SE-ResNet-50은 0.26%의 상대적 증가에 해당하는 ~3.87G FLOP을 요구하며 약간의 추가 계산 부담으로 정확도 향상과 GPU Library에 더욱 최적화되며 시간적인 측면에서도 조금 더 나은 결과를 가져옴.

SE Block에 도입된 추가 매개변수를 고려하여 두개의 FC Layer에서 발생하는 전체 네트워크 용량의 작은 부분으로 구성하며 다음 아래의 식과 같음.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75980075-26f15400-5f25-11ea-9ae3-85da4ce12f52.png" width="50%"></p>

여기서 S는 단계(공간 차원 피쳐 맵에서 작동하는 블록의 집합), Cs는 출력 채널의 치수, Ns는 장소에 대해 반복되는 블록 수를 기록함.
다음 SE-ResNet-50은 최대 250만 개의 추가 매개변수를 도입하였음.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75980081-29ec4480-5f25-11ea-949b-dbda979248d6.png" width="70%"></p>

위 표를 통해서 두 개의 FC Layer를 도입함으로써 top-1, top-5 error의 좋은 결과를 가져올 수 있었음을 보여주며 네트워크 최종 단계에서 가장 많은 수의 채널에 걸쳐 Excitation 연산이 수행되지만 비용이 많이 드는 부분도 성능 면에서 적은 비용만으로 제거될 수 있다는 것을 발견할 수 있음.

#### 5 EXPERIMENTS
다양한 작업, 데이터셋 및 모델 아키텍처에 걸쳐 SE Block 효과를 조사하기 위한 실험을 수행함.

##### 5.1 Image Classification
ImageNet 2012 dataset(train images 128만개, validation images 50K), classes_num 1000으로 검증 세트의 top-1 and top-5를 보고함.

	- Dataset
    224 x 224 pixel을 표준으로 사용하여 무작위 자르기로 데이터 확대를 수행하고 랜덤 수평 플립을 수행함.
    그리고 각 입력 이미지는 평균 RGB 채널 감소를 통해 정규화 되었음. 

	- Model Compile
    모든 모델은 대규모 네트워크의 효율적인 병렬 훈련을 처리하도록 설계된 분산 학습 시스템 ROCS에서 훈련되며 optimizers으로 동기식 SGD(momentum=0.9), batch_size=1024 사용하여 수행됨.

초기 학습 속도 0.6 설정과 30세마다 10배씩 감소, 모델 중량 초기화 전략을 사용하여 100세대에 대해 교육함.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75980095-2eb0f880-5f25-11ea-8c6a-f43e9cffbf37.png" width="70%"></p>

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75980102-32447f80-5f25-11ea-8f60-662ef812ed9e.png" width="70%"></p>

- Network depth.
SE-ResNet-50은 101-ResNet과 비슷한 성능 면에서 전체 계산 부담은 그 절반으로 SE Block 자체는 깊이를 더하지만 계산적으로는 효율적인 방식으로 기본 구조의 깊이를 확장하여 수익이 감소하는 시점에도 좋은 수익을 낸다. 이것을 통해서 서로 다른 네트워크 깊이의 범위에서 일관성이 있다는 것을 알 수 있으며 이렇게 SE Block에 의해 유도된 개선이 기본 아키텍처의 깊이를 단순히 증가시킴으로써 얻어진 개선 상황과 보완적일 수 있음을 시사함.

- Integration with modern architectures.
두 가지 최신 아키텍처인 Inception-ResNet-v2, ResNext 통합하는 효과를 연구하며 VGG-16, SE-VGG-16의 아키텍처 등 기본 넷 작업에 추가 연산 빌딩 블록을 도입함으로써 앞에서 얻은 여러가지 측면에 대해서 조사를 함으로 SE Block 최적화 절차 전반에 걸쳐 지속적으로 개선된다는 것을 관찰하며 다양한 네트워크 아키텍처에서 상당히 일관적임을 확인할 수 있었음.

- Mobile setting.
마지막으로 MobileNet와 ShuffleNet을 고려함. 이러한 실험에서는 batch_size 256, 데이터 확대 및 정규화를 사용하였음.
8개의 GPU에 걸쳐 SGD(momentum=0.9)과 10배씩 감소되는 초기 학습 속도를 사용하였으며 최대 400 epoch를 통해서 관찰할 수 있었음. TABLE 3에서 보고된 결과는 계산 비용의 최소 증가에서 지속적으로 큰 표 차이로 정확도를 향상시킨다는 것을 보여줌.

- Additional datasets.
SE Block으로 다른 데이터셋에 일반화되는지 여부를 조사함.
CIFAR-10, CIFAR-100데이터셋에서 몇 가지 인기 있는 기본 아키텍처와 SE Block을 추가한 실험을 수행함으로써 ImageNet 데이터셋에만 국한되지 않음을 확인할 수 있었음.(중간 실험 조건 부분은 생략하겠음.)

###### 이하 5.2 Scene Classification, 5.3 Object Detection on COCO, 5.4 ILSVRC 2017 Classification Competition 부분은 위와 비슷한 설명을 함으로써 생략함.

#### 6 ABLATION STUDY

SE Block 구성요소에 대해 다른 구성을 사용하는 효과를 보다 잘 이해하기 위해 실험을 수행함.
단일 머신(GPU 8개 포함), ImageNet데이터셋에서 수행, ResNet-50 아키텍처를 사용, 나머지(Data augmentation, etc..)은 위 5.1에 기술된 접근 방식을 동일하게 하며 학습 속도를 0.1로 초기화하여 실험을 함.

##### 6.1 Reduction ratio

r은 네트워크에서 SE Block의 용량과 계산 비용을 변화시킬 수 있는 하이퍼파라미터이다.
다음과 아래와 같은 실험을 하였음.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75980130-3c667e00-5f25-11ea-856b-c78bc4652dbe.png" width="50%"></p>

복잡성이 증가해도 성능은 단조롭게 개선되지 않는 반면 비율이 작을수록 모델의 매개변수 크기가 크게 증가함.
r=16일때 정확도와 복잡도 사이의 균형이 잘 잡힘. 이것은 이 아키텍처에 최적화된 비율이였으며, 주어진 기본 아키텍처의 요구를 충족시키기 위해 비율을 조정하여야 함.

##### 6.2 Squeeze Operator

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75980141-40929b80-5f25-11ea-810f-85f8a59585b2.png" width="50%"></p>

Squeeze operator로 global average pooling사용하는 것의 중요성을 보여줌. Max pooling과 Avg pooling 모두 효과적이었지만 Avg pooling이 조금 더 나은 성능을 달성하며, Squeeze 작동의 기초로서 선택을 정당화함.

##### 6.3 Excitation Operator

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75980152-438d8c00-5f25-11ea-8add-dae876b0d0d0.png" width="50%"></p>

Excitation operator로 비선형성(sigmoid)의 선택을 평가함. 그리고 두 가지 추가 옵션에 대한 부분을 고려함.(ReLU, Tanh)

##### 6.4 Different stages

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75980157-47211300-5f25-11ea-8738-460989ffc723.png" width="50%"></p>

SE Block을 ResNet-50에 단계별로 추가하여 도입될 때 성능 이점을 가져오는 것을 관찰함.
서로 다른 단계에서 성능을 더욱 강화하기 위해 효과적으로 결합될 수 있다는 점에서 보완적.

##### 6.5 Integration strategy

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75980185-53a56b80-5f25-11ea-888a-cbdf4fd10352.png" width="70%"></p>

위 그림은 SE설계에 대한 변형 부분을 시각적으로 보여줌.

SE설계에 3가지 변형을 고려함.

	- SE Block 잔여 유닛보다 먼저 이동되는 SE-PRE 
	- ID 지점(ReLU 이후)과의 합산 후 SE Unit을 이동시키는 SE-POST
	- SE Unit이 ID 연결부에 평행하게 배치되는 SE-Identity

아래 표는 다음 3가지 변형을 고려한 수행 결과를 나타냄.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75980195-57d18900-5f25-11ea-9f82-682d1914da34.png" width="50%"></p>

SE Block 블록이 각각 유사한 성능을 보이는 반면 SE-POST Block의 사용은 성능 저하를 초래한다는 것을 관찰할 수 있으며 분기 집계에 앞서 적용한다면 극적으로 좋은 성능 향상을 관찰할 수 있었음.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75980214-5ef89700-5f25-11ea-9861-b86f5dcd5a4f.png" width="50%"></p>

위의 실험에서 각 SE Block은 잔류 유닛의 구조 의부에 배치되었다. 그리고 위 표는표준 SE Block보다 더 적은 매개변수로 SE 3 x 3변량 분석 정확도를 비교할 수 있음. 비록 작업의 범위를 벗어나지만, 특정 아키텍처에 대한 SE Block 사용을 조정함으로써 축적인 효율성 이득을 달성할 수 있을거라고 예상함.

#### 7 ROLE OF SE BLOCKS

Squeeze의 작동의 상대적 중요성과 실제로 Excitation 메커니즘이 작동하는 방법을 이해하고자 함.
SE Block의 실제 기능에 대한 최소한의 이해를 달성한다는 목표를 가지고 수행하는 역할을 검토하기 위한 경험적 접근방식을 취함.

##### 7.1 Effect of squeeze

Squeeze 연산에 의해 생성된 글로벌 임베딩이 성능에 중요한 역할을 하는지 여부를 평가하기 위해, 동일한 수의 파라미터를 추가로 Global Average Pooling을 수행하지 않는 SE Block의 변형으로 실험을 함.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75980225-628c1e00-5f25-11ea-964d-7bc5705451b2.png" width="50%"></p>

글로벌 정보의 사용이 모델 성능에 상당한 영향을 미치며 squeeze작업의 중요성을 강조함.
더욱, NoSqueeze설계와 비교하여, SE Block은 이 글로벌 정보를 계산적으로 인색하게 사용할 수 있도록 함.

##### 7.2 Role of Excitation

SE Block에 있는 Excitation 연산자의 기능에 대해 SE-ResNet-50모델의 활성화를 연구하고 다양한 깊이에서 서로 다른 등급과 다른 입력 이미지에 대한 분포를 검토함. 특히 다른 클래스의 이미지에 따라, 한 클래스 내의 이미지에 따라 어떻게 변화하는지 이해하고자 함.

다음 ImageNet dataset에 관한 그래프를 보여줌과 3가지 역할에 대해 관찰함.

	- 서로 다른 등급의 분포는 네트워크 초기 계층에서 매우 유사함.(초기 서로 다른 계층에 의해 공유될 가능성이 있음.)
	- 더 깊은 곳에서 각 채널의 값은 다른 등급이 형상의 차별적 가치에 대해 서로 다른 선호도를 보일 때 훨씬 더 세분화됨.
	- 네트워크의 마지막 단계에서 다소 다른 현상(Global pooling, SE Block의 한계 손실만으로 제거함으로써 추가 파라미터 카운트를 상당히 줄임)을 관찰함.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75980234-661fa500-5f25-11ea-961d-bac21f937c00.png" width="70%"></p>

위 그래프는 SE Block 동적 거동이 클래스 내 클래스와 인스턴스 모두에 따라 다르다는 것을 나타내는 클래스 간 시각화와 일치하는 추세를 관찰함. 특히 단일 등급 내에서 표현의 다양성을 고려할 수 있는 네트워크의 후기 계층에서 네트워크는 차별적 성능을 개선하기 위해 기능 재교정을 이용하는 것을 배움. 즉, SE Block은 인스턴스별 응답을 생성하며, 아키텍처의 서로 다른 계층에서 점점 더 세분화된 모델의 요구를 지원하는 기능을 함.


#### 8 CONCLUSION

네트워크의 표현력을 향상시키기 위해 설계된 구조 단위인 SE Block을 제안함.
이는 동적 채널 측면 기능 재교정을 수행할 수 있게 함으로써 네트워크의 표현력을 향상시키기 위한 것으로 다양한 실험을 통해 성능을 달성하는 SENets의 효과를 알 수 있었다.
