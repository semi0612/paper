
## FCN(Fully Convolutional Networks for Semantic Segmentation)

논문 링크 : https://arxiv.org/pdf/1411.4038.pdf

-------------------

### Abstract
Semantic Segmentation 문제를 위해 제안된 딥러닝 모델이며, 기존 이미지 분류에서 우수한 성능을 보인 CNN 기반 모델(AlexNet, VGG 16, GoogLeNet)을 목적에 맞춰 변형시킨 것이다. 변형시킨 것에 대하여 앞으로 소개하고자 한다.

PASCAL VOC(2012년 평균 IU 62.2% 대비 상대적 개선 20%)와 NYUDv2, SIFT Flow의 최첨단 세분화를 달성하는 한편, 추론은 일반적인 이미지에 대해 1/5초 미만이 걸린다.

### Introduction

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/84623921-4bdd6600-af1b-11ea-8838-eabfa4b05661.png" width="50%"></p>

Semantic Segmentation는 pixel단위로 어떤 object인지 classification하는 것이라고 말할 수 있다.
그렇게 convolution과 pooling 과정을 반복하여 일정 영역을 반복하여 포함하는 window를 만들고 그 안 object를 분류해서 window 중앙의 pixel의 class 값이라고 간주하는 방식을 사용함에 따라 계산량과 전반적인 정보가 아닌 제한된 지역 정보만을 사용하며 fully connected layer가 고정된 크기의 입력만을 받아들이는 것과 위치 정보가 사라지는 등 여러 부족함들이 있었다.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/84623931-53047400-af1b-11ea-8f5d-1cc013d6cdbc.png" width="50%"></p>

따라서 이 논문에서는 local feature, global feature를 모두 사용하고 계산량의 큰 비율을 차지하는 fully connected layer가 없는 아키텍처를 제안한다.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/84623948-5dbf0900-af1b-11ea-8422-0f2476a53bc2.png" width="50%"></p>

따라서 네트워크는 더 이상 특정 영상 크기를 가지지 않게 된다.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/84624005-77f8e700-af1b-11ea-977f-16f2c6c3dd53.png" width="50%"></p>

마지막으로 feature map의 크기가 매우 작아졌기 때문에 원영상 크기르 복원하는 작업으로 여기서는 skip connection 방법을 사용하였다.

### Network Architecture

##### fully convolutional layers

다음 아래와 같은 Fully Convolutional Networks(Convolution + Pooling)의 구조를 지니며, 크게 4가지 부분으로 구성됨.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/84619916-775b5300-af11-11ea-96c5-566170fb8213.png" width="50%"></p>

위 그림을 통해 AlexNet의 뒷단에 사용되던 3개의 fully connected layer를 1 x 1 convolution으로 간주한 경우를 보여주며, 이를 convolutionization이라고 부른다. 그래서 여기서 1 x 1 convolution은 기존처럼 위치 정보가 사라지는 것이 아니며 위 heatmap과 같은 위치의 score값을 확인할 수 있다.

간단하게 요약하자면, fully connected layer를 1 x 1 convolution으로 간주함에 따라 위치 정보(공간 정보)를 유지할 수 있게 되었고, 전부 convolutional network으로 구이 되기 때문에 입력 영상의 제한을 받지 않는다. 또한 patch단위로 영상을 처리하는 것이 아닌 전체 영상을 한꺼번에 처리할 수 있어서 겹치는 부분에 대한 연산을 줄일 수 있다.

##### Upsampling(Deconvolution)
Upsampling이전 shift-and-stitch 방식을 이용하는 것에 대해서 검토를 하였다. 그렇지만 Upsampling을 사용하는 것이 더 효과적이라는 것으로 결론을 함으로써 사용하지 않았다.

다음 아래 그림은 fully convolutional Network 아키텍처 구조이며 간단하게 보면 feature map의 크기가 줄어들고 커지는 것을 볼 수 있다.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/84618138-39a7fb80-af0c-11ea-8344-3938d0c30ed7.png" width="50%"></p>

1 x 1 convolution을 거치면서 얻어진 score 값을 원 영상의 크기로 확대하는 가장 간단한 방법은 bilinear interpolation을 사용하면 됨.

end-to-end 학습의 관점에서는 고정된 값을 사용하는 것이 아니라 학습을 통해서 결정하는 편이 좋다.

즉, deconvolution에 사용하는 필터의 계수는 학습을 통해서 결정되며 경우에 따라서 bilinear 필터를 학습할 수도 있고 non-linear upsampling도 가능하게 된다.

그렇지만 score를 upsampling하게 되면, 성능의 부분을 기대하기가 어려워진다. 그래서 이 부분을 개선시키고자 다음 아래 그림과 같이 'skip layer' or 'skip connection' 개념을 활용한다(두 개념 동일).

다음 아래 그림과 같이 특정 한개만의 feature map(score)만을 사용하는 것이 아니라 다른 크기를 갖는 feature map의 값도 같이 사용하는 방식을 가졌다.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/84625233-d3c46f80-af1d-11ea-966b-6d0278b89536.png" width="60%"></p>

이것을 'deep jet'이라고 하며, 이전 layer 일수록 세밀한 특징을 갖고 있기 때문에 이것을 합하면 보다 정교한 예측이 가능해진다.

따라서 다음 그림과 같은 과정으로 이루어진다.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/84621937-c9eb3e00-af16-11ea-8c08-1c5be10fc7ee.png" width="60%"></p>

여러 단계를 거치면서 feature map의 크기가 너무 작아지면 섬세한 부분이 많이 사라지게 되기 때문에 최종 과정보다 앞선 결과를 사용하여 섬세함을 보강하는 것이다.

조금 더 자세하게 확인하면 다음 아래 그림과 같다.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/84626081-73cec880-af1f-11ea-865d-e543b44b116a.png" width="60%"></p>

FCN-32s 1/32에서 32배만큼 upsample한 결과이며, FCN-16s는 아래 그림과 같이 pool5의 결과를 2배 upsample한것과 pool4의 결과를 합치고 다시 그 결과를 16배 upsample하는 것을 볼 수 있다. 이후 과정은 생략함.

따라서 위의 과정 여러 단계를 합쳐주는 과정을 거치면 다음 아래 그림과 같이 더 섬세하고 정교한 예측이 가능해진다.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/84622358-c1dfce00-af17-11ea-969d-668aaccccd5c.png" width="50%"></p>

위 예시에서 stride 32와 8 부분을 보면 확연하게 차이점에 대해서 볼 수 있다.

### Results
Semantic segmentation은 대표적으로 장면 이해(scene understanding)을 위한 기반 기술로 로봇 비전으로 장면을 이해하는데 사용될 수 있고, 자율 주행의 영상 부분에도 적용이 가능하다.

그래서 이 논문에서 연구한 실험의 결과적으로는 다음 아래와 같이 FCN의 성능을 확인할 수 있다.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/84622750-b640d700-af18-11ea-8783-36ee7fee4ee4.png" width="50%"></p>

Ground Truth 값과 FCN의 값을 비교하면 확연하게 차이점에 대해서 볼 수 있다. 결과적으로 간단하게 한 줄로 요약하자면, Classification Network의 뒷부분에 있는 fully connected layer를 1 x 1 convolution으로 간주하는 것을 기반으로 속도 및 성능을 얻을 수 있게 되었다.

그리고 PASCAL VOC 2011 데이터를 이용하여 실험한 결과 아래 표와 같다.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/84626394-f9527880-af1f-11ea-9ea4-10292da3fa69.png" width="50%"></p>

FCN-32s 결과와 FCN-8s의 결과를 보면 정밀한 예측 개선이 되는 것을 확인할 수 있다.
