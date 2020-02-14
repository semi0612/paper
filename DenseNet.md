# Densely Connected Convolutional Networks(DenseNet)

논문 링크 : https://arxiv.org/pdf/1608.06993.pdf

## 논문 요약

> 1. 2016년 논문을 통해 발표된 CNN모델
> 2. ResNet의 Skip connection과는 다른 Dense Connetivity를 제안
> 3. 이 모델의 장점 (1) 이미지에서 저수준들의 특징이 잘 보존된다
>                   (2) gradient가 수월하게 흘러 gradient vanishing 문제가 발생하지 않는다
>                   (3) 깊이에 비해 파라미터 수가 적기에 연산량 절약 + 적은 데이터 셋에서도 비교적 학습이 잘 된다


## 특징
> ResNet 은 입력에 출력이 더해지는 것이라 후에
