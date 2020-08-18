## Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization

논문 링크 : https://arxiv.org/pdf/1610.02391.pdf

### summary
```
* (GAP (Global average pooling) : feature map 전체에서 pooling
* Grad-CAM과 CAM은 'CNN모델'이 'GAP 레이어'를 사용한다는 가정하에 동일한 동작.
* CAM은 GAP가 있어야만 사용이 가능하고, GAP는 채널에 대한 평균이기에 CNN에서만 사용 가능

```
-------------------



### ABSTRACT
CNN의 많은 클래스로부터 결정을 위한 '시각적 설명'을 제작하면서, 보다 투명하고 알기 쉬운 기법으로 
이전 접근 방식과는 달리 Grad-CAM은 다양한 CNN Model에서 활용할 수 있는 장점이 있음

#### 1. INTRODUCTION
Grad-CAM은 아키텍처 변경이나 재교육 없이 CNN 기반 네트워크에 대해 시각적 설명을 생성하는 기법으로, 이미지 분류에 대해 시각화는, 현재의 CNN 실패에 대한 통찰력을 제공
훈련되지 않은 사용자가 '강력한' 네트워크와 '약한' 네트워크를 성공적으로 식별 할 수 있도록 돕는다
![캡처](https://user-images.githubusercontent.com/51469989/82171412-310ad800-9902-11ea-9b25-5867154e3ea8.JPG)

#### 2. Related Work
이전 연구들의 경우 이미지를 세밀하게 시각화 하지만 class discriminative 하지는 않음.
또, CAM은 Softmax 바로 직전 특징맵이 필요하기 때문에 GAP가 있는 구조에만 적용할 수 있다는 한계가 있다면
Grad-CAM은 single forward 기반이기에 효율적이고 더 많은 곳에 사용이 가능

#### 3. Grad-CAM
Grad-CAM 및 Guided Grad-CAM에 대한 접근 방식을 제안한다.
아래 그림에서 볼 수 있듯 Grad-CAM을 표현하기 위해서는 Conv 층의 Gradient를 활용하여 해당 그래디언트와 feature maps에 곱연산하고 추가적으로
ReLU를 거친 값을 입력이미지에 heat-map 해줘야 하는 걸 알 수 있다.
![캡처](https://user-images.githubusercontent.com/51469989/82644102-b61c2700-9c4b-11ea-8c1e-f29b1f9f1594.JPG)


##### 3.1 Grad-CAM generalizes CAM
![캡처](https://user-images.githubusercontent.com/51469989/82658248-a3145180-9c61-11ea-96d6-69f02cce8e75.JPG)
![캡처](https://user-images.githubusercontent.com/51469989/82658315-bd4e2f80-9c61-11ea-98ad-96138f011c1e.JPG)

위 식을 통해 결국 Grad-CAM은 CAM을 일반화 시킨 것임을 할 수 있다. Grad-CAM은 CAP의 제약이 없기 때문에 Feature maps의 채널 수 만 일치한다면 어떤 layer든 시각화가 가능하다.
##### 3.2 Guided Grad-CAM
Guided-Backpropagation과 Grad-CAM을 융합, class-discriminative 하면서
fine-grained importance를 시각화하는 방법을 제안 함.
먼저 Guided Backpropagation을 이용해서 saliency map을 구한 후,
Grad-CAM과 saliency map의 크기를 bi-linear interpolation 을 사용해서 동일하게 맞춰주고 서로 곱해준다.

#### 4. Evaluating Localization Ability of Grad-CAM
기존의 VGG-16, AlexNet, GoogleNet을 통해 Grad-CAM의 localization 능력을 영상 분류의 맥락에서 평가했을때,
Grad-CAM의 localization 오류는 c-MWP의 오류보다 덜하다. 
![캡처](https://user-images.githubusercontent.com/51469989/82647261-e7e3bc80-9c50-11ea-8ef0-792489f42b71.JPG)

#### 5. Evaluating Visualizations
대량의 이미지 분류에 대한 매우 깊은 경직망(최대 19개)을 평가하며 표현 깊이가 분류 정확도에 이롭다는 것이 입증되었으며,
깊은 Convnet 아키텍처를 사용하여 ImageNet 챌린지 데이터 세트의 높은 성능을 달성할 수 있었다.

#### 6 Diagnosing image classification CNNs with Grad-CAM
이미지 분류 CNN 진단 및 데이터 세트의 바이어스 식별과 같은 Grad-CAM의 특정 사용 사례를 보여준다.

##### 6.1 Analyzing failure modes for VGG-16
![캡처](https://user-images.githubusercontent.com/51469989/82653602-6db83580-9c5a-11ea-8173-6c5af83b3608.JPG)

네트워크가 어떤 실수를 저지르고 있는지 확인하기 위해, 먼저 네트워크가 (여기서는 VGG-16) 정확하게
분류하지 못하는 예시를 받아, 모두 시각화 한다.

##### 6.2 Effect of adversarial noise on VGG-16
![캡처](https://user-images.githubusercontent.com/51469989/82654088-27afa180-9c5b-11ea-8931-6608c93355e3.JPG)

(a-b) "항공기" 범주에 대한 원본 이미지와 생성된 적대적 이미지
(c-d) 원래 범주 "tiger cat" 및 "boxer (dog)"에 대한 Grad-CAM 시각화

네트워크가 완전히 속아 높은 신뢰도로 지배적인 카테고리 라벨을 예측을 함에도 불구하고
Grad-CAM은 원래의 카테고리를 정확하게 localization 할 수 있다.

##### 6.3 Identifying bias in dataset
교육 데이터 세트의 편향성을 식별하고 감소시킬 수 있다.
모델이 의사와 간호사를 구별하기 위해 사람의 얼굴/머리 스타일을 본다는 것을 밝혀내 편향된 예측을 바로 잡을 수 있었다.

![캡처](https://user-images.githubusercontent.com/51469989/82656890-70695980-9c5f-11ea-9188-a14779c49725.JPG)

첫번째 줄에서 편향된 모델(모델1)은 그 사람이 간호사인지 여부를 결정하기 위해 그 사람의 얼굴을 보고 있는 반면, 편향되지 않은 모델은 결정을 내리기 위해 짧은 소매를 보고 있다는 것을 알 수 있다
두 번째 행 역시 얼굴과 헤어스타일을 보고 예측(의사를 간호사로 잘못 분류)한 반면, 편향되지 않은 모델은 하얀 코트와 청진기를 보면서 올바른 예측을 했다.


#### 7. Textual Explanations with Grad-CAM
GradCAM으로 텍스트 설명을 얻을 수있는 방법을 제공 

![캡처](https://user-images.githubusercontent.com/51469989/82657694-bd016480-9c60-11ea-812f-240f264c9e0e.JPG)

뉴런들을 이용해 마지막 층의 이름을 얻는다. 상위 5개 뉴런과 하위 5개 뉴런의 등급별 중요도 점수인 αk를 기준으로 분류, 이들은 텍스트 설명으로 사용 할 수 있다.
텍스트 설명을 위해 예측된 클래스에서 가장 중요한 뉴런을 이름과 함께 제공하는데, 이는 잘못 집중하고 있는 부분을 확인할 수 있게끔한다.
처음 두 행은 성공 사례를 마지막 두행은 실패한 사례를 각각 보여주고 있다.

#### 8. Grad-CAM for Image Captioning and VQA
Grad-CAM을 비전 및 언어 모델 (이미지 캡션 및 시각적 질문 응답 (VQA))에 적용하는 방법을 보여준다.
결과적으로 Grad-CAM을 이미지 captioning 및 VQA와 같은 언어작업에 적용. 이는 눈에 띄게 변하지 않는 기존의 시각화와 비교된다. 
