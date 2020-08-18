## EfficientNet (Rethinking Model Scaling for Convolutional Neural Networks)

논문 링크 : https://arxiv.org/pdf/1905.11946.pdf

### summary
```
* compound scaling 을 잘 해보겠다는 것이 목적이다.
* 직관적으로 세가지(depth, channel width, resolution)요소를 scale_up하여
  compound scaling을 적절하게 구성, 컨볼루션 네트워크의 성능을 높이고자 하였다.
```
-------------------

### Abstract
모델 스케일링을 체계적으로 연구하여 네트워크 Depth, Width 및 Resolution의 세심한 균형이 더 나은 성능으로 이어질 수 있음. 그럼으로 복합 계수를 사용하여 모든 차원의 Width/Depth/Resolution를 균일하게 확장하는 새로운 스케일링 밥법을 제안.

EfficientNet-B모델 ImageNet에서 84.4% top-1 / 97.1% top-5 정확도를 달성함.

결과적으로 높은 정확도와 매개변수가 크게 감소하는 효율성을 얻음.

#### 1. Introduction
ResNet, Gpipe 등 여러 모델에서 ConvNets의 크기를 증가시킴으로써 더 나은 정확성을 얻을 수 있었다. 그리고 Depth, Width, Resolution 등 3차원 중 하나만 스케일링 하는 것이 일반적이었으며 임의의 스케일링은 지루한 수동 조정이 필요하며 최적 이하의 정확성과 효율성을 산출하는 경우가 많았음.

지금까지 정확성과 효율성을 달성할 수 있는 'ConvNets scale up 원칙적인 방법이 있었나?'라는 의문점이 생기면서 이번 연구에서는 경험을 바탕으로 Width/Depth/Resolution의 모든 차원의 균형을 일정한 비율로 단순히 확장하여 해결할 수 있었음.
그래서 지금부터는 복합 스케일링 방법을 제안으로 고정 스케일링 계수의 집합으로 네트워크 Depth, Width 및 Resolution를 균일하게 스케일링함.

Ex) 2^N배 더 많은 계산 자원을 사용한다면 네트워크 Width(a^N), Depth(b^N), Resolution의(r^N)으로 증가시킬 수 있음. 

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75366426-0eb87e00-5902-11ea-84ee-fb65ccd6b35c.png" width="50%"></p>

위 그림은 기존 스케일링 방법과 위에서 설명한 복합 스케일링 방법에 대해서 한눈에 확인이 가능함. 
직관적으로 복합 스케일링 방법은 입력 이미지가 더 큰 경우 수용 필드를 증가시키기 위해 더 많은 레이어가 필요하고 세밀한 패턴을 캡처하기 위해서 더 많은 채널이 필요하기 때문에 이는 합리적임을 보여줌. 그리고 모델 확장의 효율성은 baseline network에 따라 크게 달라짐.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75366437-11b36e80-5902-11ea-9737-638eb491827b.png" width="50%"></p>

결과적으로 위 그래프를 확인하면 다른 모델과 비교 하였을때 적은 매개변수로 높은 정확도로 얻을 수 있음을 보여줌.

#### 2. Related work

- Convnet Accuracy
AlexNet, GoogleNet, SENet 등 ImageNet, ConvNet 좋은 결과를 달성할 수 있었음. 그리고 GPipe는 네트워크를 분할하고 각 부품을 다른 가속기로 분산시켜 전문화된 파이프라인 병렬 라이브러리로만 교육할 수 있을 정도로 다양한 전송 학습 데이터와 컴퓨터 비전 작업에서 더 나은 성능을 가져옴.
비록 많은 애플리케이션에 더 높은 정확도가 필요하지만 이미 하드웨어 메모리 한계에 도달 했고, 따라서 더 많은 정확도 향상은 더 나은 효율을 필요로 했음.

- ConvNet Efficiency
Deep ConvNets은 지나치게 매개변수를 사용함.
모델 압축은 효율을 위해 정확도를 거래함으로써 모델 크기를 재조정하는 방법을 소개하고자 함.
유비쿼터스화되면서 SqueezeNets, MobileNets, ShuffleNets 등 효율적인 모바일 사이즈의 ConvNets 설계에서 네트워크 Width, Depth 컨볼루션 커널 유형 및 크기를 광범위하게 조정함으로써 모바일 ConvNets보다 훨씬 더 높은 효율을 달성함.
따라서 본 논문에서는 최첨단 접근성을 능가하는 초대형 ConvNets의 모델 효율을 연구하는 것을 목표로 함.

- Model Scaling
더 나은 효율과 정확성을 달성하기 위해 네트워크 Depth, Width, Resolution 3차원 모두에 대한 ConvNet Scaling을 체계적이고 경험적으로 연구하고자 함.

#### 3. Compound Model Scaling
스케일링 문제를 공식화하고, 다양한 접근 방법을 연구하며, 새로운 스케일링 방법을 제안함.

##### 3-1. Problem Formulation
ConvNet Layer i는 다음과 같은 함수로 정의함.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75366446-1546f580-5902-11ea-8339-c1831b3b6257.png" width="50%"></p>

Yi = Fi(Xi)는 연산자로 출력 텐서, Xi는 입력 텐서이며, Fi를 나타내는 Li를 반복하는 경우 <Hi, Wi, Ci>는 층 i의 입력 텐서 X의 모양을 나타냄.
따라서 모델 스케일링은 기준선 네트워크에 미리 정의된 Fi를 변경하지 않고 네트워크 Depth(Li), Width(Ci), Resolution(Hi, Wi)를 확장하려고 함.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75366452-17a94f80-5902-11ea-8762-fa09cce83647.png" width="50%"></p>

위 에서 Fi, Li, Hi, Wi, Ci는 기준석 네트워크에서 사전 정의된 파라미터를 보여줌.
설계 공간을 더욱 줄이기 위해 모든 레이어를 일정한 비율로 균일하게 스케일링해야 한다고 제한함. 따라서 최적화 문제로 공식화할 수 있는 주어진 자원 제약에 대한 모델 정확도를 극대화하는 것을 목표로 함.

##### 3-2. Scaling Dimensions
Depth(d)/Width(w)/Resolution(r) 서로 의존하고 다른 자원 제약 하에서 값이 변화한다는 것에 대해서 다음과 같은 차원 중 하나로 ConvNets을 확장함.

- Depth(d)

Scaling network depth은 많은 ConvNets가 사용하는 가장 일반적인 방법으로 심층적인 ConvNet이 더 풍부하고 더 복잡한 특징을 포착할 수 있고, 새로운 작업에 대해 잘 일반화할 수 있음. 또 지금까지 매우 깊은 ConvNets에 대한 정확도 하락을 보여줌.

- Width(w)

- 아래 그림으로는 Width(w), Depth(d), Resolution(r) 순으로 FLOPS에 대한 top-1 정확도를 보여줌.(FLOPS(Floating Point Operations Persecond) - 컴퓨터의 성능을 수치로 표현하는 단위, 1초동안 수행할 수 있는 부동소수점 연산의 횟수)

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75366460-1aa44000-5902-11ea-95eb-9e7e0640fb79.png" width="70%"></p>

넓은 네트워크는 보다 세분화된 특징을 포착할 수 있는 경향이 있으며, 훈련하기가 더 쉬우며 얕은 네트워크는 더 높은 수준의 특징을 포착하는데 어려움을 겪음.

- Resolution(r)

더 높은 해상도의 입력 영상을 통해 ConvNets은 더 세밀한 패턴을 포착할 수 있었으면서 기존 224 x 224 해상도를 시작으로 보다 정확한 정확성을 위해299 x 299, 331 x 33, Gpipe 480 x 480, ImageNet 600 x 600 을 통해 달성할 수 있었음. 또한 매우 높은 해상도의 경우 정확도는 감소하였음.

위의 분석을 통해 첫 번째 관찰로 이어짐.

###### - Observation 1 - 3차원을 확장하면 정확도는 향상되지만 대형 모델의 경우 정확도가 감소함.

##### 3-3. Compound Scaling

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75366467-1ed05d80-5902-11ea-8939-ddb85ca964da.png" width="50%"></p>

기존의 단일 차원 스케일링보다는 서로 다른 스케일링 치수를 조정하고 균형을 맞출 필요가 있음을 보여줌으로 다음 그래프는 Depth, Resolution에서 1.0보다 2.0에서 높은 정확도를 보여줌.

###### -  Observation 2 - 더 나은 정확성과 효율성을 추구하기 위해서는 ConvNet 확장 중에 네트워크 Width, Depth 및 Resolution의 모든 차원의 균형을 맞추는 것이 중요함.

Compound scaling method 제안

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75366472-2263e480-5902-11ea-8bc9-3d0782aa200f.png" width="50%"></p>

Reqular convolution op의 FLOPS는 d, w^2, r^2의 비례함. 즉, 네트워크 Depth를 두배로 하면 FLOPS가 두배가 되지만, 네트워크 폭이나 해상도를 2배로 하면 FLOPS가  4배가 증가함.
따라서 위의 식을 보면 총 FLOPS가 대략 (a * p ^2* r^2)자승 만큼 증가함.

#### 4. EfficientNet Architecture
좋은 기준선 네트워크를 갖는 것도 중요하며 정확도와 FLOPS를 모두 최적화하는 다중 객체 신경 구조 검색을 활용하여 기준 네트워크를 개발함. 구체적으로는 acc(m) * [FLOPS(m)/T]^w을 최적화 목표로 사용함. 이것은 하드웨어 장치를 대상으로 하는게 아니기 때문에 지연 시간이 아닌 FLOPS을 최적화하는 것으로 함.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75366481-25f76b80-5902-11ea-869b-7d49824a1f22.png" width="50%"></p>

위 표는 mobile inverted bottleneck MBConv이며 여기에는 queeze-and excitation optimization이 추가적으로 구성됨.
기본 EfficientNet-B0에서 시작하여 복합 스케일링 방법을 적용하여 2단계로 확장함.
첫번째 단계로 EfficientNet-B0에 대한 제약 α = 1.2, β = 1.1, γ = 1.15  하에 최선의 값을  α · β 2 · γ 2 ≈ 2 가짐.
두번째 단계로 세가지 계수를 고정하고 서로 다른 자승을 기준선 네트워크를 Scale up하여 EfficientNet-B1에서 B7까지 획득 함.

#### 5. Experiments

EfficientNets에 대한 스케일링 방법으로 여러 데이터셋과 네트워크를 비교하여 평가함. - 이 부분은 따로 논문 참고 바람.

EfficientNets가 실제 하드웨어면에서도 빠름을 Gpipe와 비교하여 보여줌.

아래 그림은 Class Activation Map(CAM)을 보여줌.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75366492-298af280-5902-11ea-81e8-5ebe7118fdec.png" width="70%"></p>

보여주고자 하는 것은 복합 스케일링 함수를 통해 스케일링 모델은 객체 상세도가 더 높은 관련 영역에 집중할 수 있다는 것이다.

결과적으로는 다른 ConvNets보다 훨씬 적은 매개변수와 FLOPS로 더 나은 정확도를 달성할 수 있었음.

#### 6. Discussion
복합 스케일링(d, w, r)의 중요성을 한번 더 부각하며 개체 세부 정보가 더 많은 관련 영역에 초점을 맞추는 경향이 있는 반면, 다른 모델은 개체 세부 정보가 부족하거나 이미지의 모든 개체를 캡처할 수 없음.

#### 7. Conclusion
네트워크 Width, Depth 및 Resolution의 균형을 신중하게 맞추는 것이 중요하지만 누락된 부분을 확인함으로써 더 나은 정확성과 효율성을 방해 하였음. 그래서 이 부분은 복합 스케일링 방법을 통해 보완할 수 있었다.
