# Very Deep Convolutional Networks For Large-Scale Image Recognition(VGG)

논문 링크 : https://arxiv.org/pdf/1409.1556.pdf

## 논문 요약

> 1. ILSVRC 2014 대회에서 GoogleNet에 근소한 차이로 아쉽게 2등을 차지한 Network
> 2. VGG Net 논문은 망의 깊이가 딥러닝 결과에 어떤 영향을 미치는 지 알아보기 위하여 연구되었다
> 3. 3x3 filter size 사용
> 4. VGGnet, GoogleNet 이전의 depth는 8layers 수준에서 머물렀다면 GoogleNet, VGGnet 이후 크게 깊어졌다
> 5. 깊은 네트워크를 가지고 있지만, GoogLeNet과 비교하면, 구조가 매우 간단하다는 장점
> 6. 매우 많은 파라미터를 이용하여 연산한다는 단점이 있다.

## 특징
> VGG 이전의 Convolutional Network를 활용한 이미지 분류 모델들은 비교적 큰 Receptive Field를 가진 11x11 또는 7x7 size의 Convolution Fiter를 사용하고 있었다. 하지만 VGG Model에서는 기존에 사용하던 것보다 작은 3x3 size의 filter를 사용하면서, 6개의 깊이가 다른 (layer depth ) Network를 구성해 실험하였다. 
> Conv filters에 3x3 filter size를 스택구조로 쌓아 여러개(깊게) 사용하는 것이 5x5 혹은 7x7 size의 filter를 사용하는 것과 비슷한 효과를 내면서도 파라미터 수는 조금 더 줄이고, 이미지 분류 정확도는 개선시키는 결과를 보여주었다.

