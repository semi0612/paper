## Very Deep Convolutional Networks For Large-Scale Image Recognition(VGG)

논문 링크 : https://arxiv.org/pdf/1409.1556.pdf

## 논문 요약

> 1. ILSVRC 2014 대회에서 GoogleNet에 근소한 차이로 아쉽게 2등을 차지한 Network
> 2. VGG Net 논문은 망의 깊이가 딥러닝 결과에 어떤 영향을 미치는 지 알아보기 위하여 연구되었다
> 3. 3x3 filter size 사용
> 4. VGGnet, GoogleNet 이전의 depth는 8layers 수준에서 머물렀다면 GoogleNet, VGGnet 이후 크게 깊어졌다
> 5. 깊은 네트워크를 가지고 있지만, GoogLeNet과 비교하면, 구조가 매우 간단하다는 장점
> 6. 매우 많은 파라미터를 이용하여 연산한다는 단점이 있다.

## 특징
> VGG 이전의 Convolutional Network를 활용한 이미지 분류 모델들은 비교적 큰 Receptive Field를 가진 11x11 또는 7x7 size의 Convolution Fiter를 사용하고 있었다. 하지만 VGG Model에서는 기존에 사용하던 것보다 작은 3x3 size의 filter를 사용하면서, 6개의 깊이가 다른 (layer depth) Network를 구성해 실험하였다. 

> Conv filters에 3x3 filter size를 스택구조로 쌓아 여러개(깊게) 사용하는 것이 5x5 혹은 7x7 size의 filter를 사용하는 것과 비슷한 효과를 내면서도 파라미터 수는 조금 더 줄이고, 이미지 분류 정확도는 개선시킴.

## 실험 내용

> 실험 모델 사이의 layer 및 파라미터 수 비교   
본 논문에서 학습시킨 ConvNet들의 설정을 볼 수 있다. 입력 이미지의 경우 ImageNet의 Dataset이기 때문에 모두 동일. 네트워크의 깊이가 깊어짐에도 불구하고(A->E 순서로 깊어짐), parameter수가 급격히 늘어나지 않음을 알 수 있다.
<img width="835" alt="vgg_1" src="https://user-images.githubusercontent.com/51469989/74997722-38a11900-549a-11ea-957e-7f60295ac39f.png">

> single test scale 모델간의 성능 비교   
깊이가 깊어질 수록 에러율 감소(단, 19레이어의 경우는 예외)
<img width="967" alt="vgg_2" src="https://user-images.githubusercontent.com/51469989/74998425-0d1f2e00-549c-11ea-8f53-56d9efae348e.png">

> multi test scale 모델간의 성능 비교   
single scale보다 multiple scale 에러율이 더 낮음을 확인할 수 있다.
<img width="1022" alt="vgg_3" src="https://user-images.githubusercontent.com/51469989/74998432-101a1e80-549c-11ea-92c5-7b8248c50b38.png">   


## 결론
> VGG_Net은 기존 CNN 구조와 마찬가지로 마지막에 Fc layer가 오는, 1개 이상의 convolutional layer 뒤에 max-pooling layer 가 오는 단순한 구조로 되어 있다. A 모델과 A-LRN 모델의 차이인 LRN layer의 유무에 따른 모델 성능을 비교한 결과 따로 성능 향상은 없다고 판단. B 모델 ~ E 모델은 layer의 깊이에 따라 모델 성능이 있는가를 비교 하였다. 그 결과 망이 깊어질수록 성능 향상을 있음을 확인 하였고, multi-crop과 dense를 섞어 사용한 multi test scale을 사용한 모델들이 조금 더 높은 성능을 보였다.   

> 기존 Model들 보다 작은 3x3 filter size를 사용하여 최대 19 레이어까지 깊이를 늘려 좋은 성능을 이끌어냈다. 하지만 여전히 paramater 수가 너무 많은데, 이는 과적합등의 문제를 발생 시킬 수 있다는 의미이기도 하다.
