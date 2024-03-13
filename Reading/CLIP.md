	> LMM : Large Multimodal Model
	> CLIP은 GPT4나 LLaVA등의 시초 연구
	  기존의 컴퓨터 비전 모델들은 주로 이미지만을 학습하여 성능을 향상. 하지만 이러한 접근 방식은 모델의 강건함과 일반화 능력에 한계를 가지고 있을 수 밖에
	  반면 LLM들은 급속도로 발전해왔기에, 이미지 인식 분야에서도 언어모델처럼 대규모 데이터셋을 학습하는 방식이 이미지 인식 분야에서도 중요한 역할을 할 수 있다고 생각
	> 기존방법
		⇨ Vision Model : 전통적으로 이미지를 입력받아 어떻게 모델을 구성하면 더 좋은 표현을 학습하는지를 고민
						 Inception[1], ResNet[2] 등은 효율적인 모듈을 구성하여 깊은 모델을 만드는 방법을 고민했고
						 SENet[3], BAM[4], CBAM[5] 등은 Attention 모듈을 적용하는 방법을 제안했음
						 또 하나의 트랜드는 Transformer[8] 구조를 적용하는 것
						 이런식으로 Vision Model은 날이 갈수록 발전해갔으나, 이미지만 학습한 모델은 고질적으로 일반화 능력이 부족하고 작은 노이즈에도 취약한 약점을 보였음
		⇨ Language Model : 2017년 Transformer[8]의 발표를 기점으로 큰 변화가 생김
						   전통적인 seq2seq 방식의 한계를 뛰어넘어 긴 문장도 효과적으로 처리할 수 있게 되었기 때문
						   이후 GPT 시리즈들 (GPT-1[9], GPT-2[10], GPT-3[11]), BERT[12] 등 다양한 초거대언어모델 (LLM)이 발표되면서 LLM의 시대가 도래함
	> 이렇게 LLM이 급격하게 발전하는 것을 보면서 Vision Model도 LLM과 같은 방향으로 간다면 한 단계 더 발전할 수 있지 않을까 생각을 함
	  LLM에 있는 두가지 조건, 큰 모델과 큰 데이터
	  그러니 일단 데이터셋이라도 아주 크게 만들어서 학습한다면 지금의 Vision Model의 한계를 넘어설 수 있지 않을까
	  당시 Vision Model의 대표 데이터셋은 ImageNet 이었는데.
	  데이터의 양이 적은편은 아니지만, 각 이미지에 대해 사람이 직접 Label을 달아놓은 형태의 데이터셋으로 이는 아무래도 갯수의 한계가 있을 수 밖에 없었음
	
	🔆데이터 셋
	> CLIP 저자들은 우선 대용량의 데이터셋을 확보하기 위해 인터넷에서 데이터를 모음. 이렇게 수집된 데이터는 이미지별 Label이 없다는 단점이 존재
	  이를 해결하기 위해 자연어를 Supervision으로 사용하는 방법을 선택함
	  (Supervision은 지도, 지시, 감독 등의 뜻을 갖는 단어로 모델에게 이미지를 설명해주는 역할. 기존 ImageNet으로 학습하는 모델들에게는 Label 정보가 Supervision)
	  즉, 인터넷상에서 이미지마다 달려 있는 자연어 문장을 그대로 Supervision으로 사용하자는 아이디어
	  
	  이 방법론의 성능을 30개의 다른 벤치마크 데이터셋을 이용해 조사했으며, 그 예로는 아래와 같다.
	  OCR, action recognition in videos, geo localization,
	  many types of fine-grained object classification(자동차 모델, 새 종류, 개종류 같이 구분하기 어려운 클래스들을 분류하는 과제)
	  
	  CLIP은 이런식으로 4억장의 이미지-자연어 매칭 데이터 셋을 구축
	  

	🔆학습방법
	> 이렇게 만든 대용량 데이터 셋은 기존 데이터셋처럼 Cross Entropy Loss를 이용해 학습 할 수가 없었음.
	  왜냐하면 자연어는 Label과 달리 특정 개수로 구분되지 않기 때문에. 따라서 softmax로 구분하는 방식의 방법은 가능하지 x
	  CLIP 저자들이 선택한 방법은 Contrastive Learning
	  Contrastive의 ‘대조하는’ 이라는 뜻처럼, 매칭되는 데이터 Feature들끼리는 가까워지도록 나머지 Feature들 끼리는 멀어지도록 학습하는 방법
	  즉 데이터를 대조해가며 나랑 매칭되는 데이터는 가까워지도록, 다른 데이터는 멀어지도록 모델을 학습하는 방법임.
	  Contrastive Learning 학습 방법은 Self Supervised Learning에서 그 진가를 발휘했는데, Label 정보가 없어도 어떠한 기준으로 나와 매칭되는지만 설정해주면 학습을 할 수 있어서
	  Contrastive Learning을 사용한 대표적인 Self Supervised Learning 모델은 SimCLR[13]가 있음
	  입력 이미지에 Augmentation을 적용하여 동일한 이미지 버전끼리는 가까워지도록, 다른 이미지 버전과는 멀어지도록 학습
	  놀라운건 이렇게 Label 정보 없이 학습했음에도 Label 정보로 학습한 모델에 버금가는 표현력을 학습했음을 실험적으로 증명했다고

![image](https://github.com/semi0612/paper/assets/51469989/292df233-27fe-4fab-818a-5eebbe8d8e6a)

	  그림을 보면
	  이미지는 Image Encoder로, 자연어는 Text Encoder로 Feature를 추출해준 후
	  추출한 Image Feature는 초록색 사각형(IN)으로, Text Feature는 보라색 사각형(TN)으로 표현 (N은 배치 개수)
	  총 N개의 Image Feature가 있고, 마찬가지로 N개의 Text Feature가 추출되어 있는 상황으로 이들 각각을 매칭하면 총 NxN개의 조합이 나옴
	  Contrastive Learning은 나와 매칭되는 조합은 가까워지도록, 그 외의 조합은 멀어지도록 학습하는 방법이라고 했는데
	  이때 가까워진다는 의미는 여기서는 두 Feature의 Cosine Similarity가 커지는 방향을 말하는 것이다.
	  두 개의 Feature가 공간상에서 가까운 각도에 위치할수록 Cosine Similarity는 큰 값을 가지도록하고, 나머지 쌍과는 멀어지도록 모델을 학습
	  여기서 모델은 Image Encoder와 Text Encoder를 의미
	  
	  결국 CLIP 방법론의 핵심은 Image Encoder와 Text Encoder를 Contrastive Learning 방법으로 학습한다는 것인데
	  Image Encoder는 다양한 Vision Model들이 가능
	  대표적으로 ResNet[2], ViT[7]가 있음.
	  ResNet[2]은 조금 더 표현 추출 능력을 강화하기 위해 마지막 Global Average Pooling 부분을 수정 후, Attention 모듈을 추가한 Attention Pooling으로 적용
	  ViT[7]는 기존 구성 거의 그대로 사용
	  이런식으로 수정된 5가지 종류의 ResNet(4x, 16x, 64x of ResNet-50, denoted as RN50x4, RN50x16, RN50x64)과 3가지 종류의 ViT를 사용하여 실험을 진행
	  Text Encoder는 마지막 Token에서 추출된 Feature를 Linear Projection 해주어 Image Feature와의 차원을 맞춰 준 Transformer[8]를 사용하였음
	
	🔆특이한점(특징)
	> Zero Shot Prediction이 가능
	  Zero Shot Prediction이란 말 그대로 한번도 학습하지 않은 문제를 맞추는 방법으로
	  기존 ImageNet으로 학습한 모델에서는 학습하지 않은 클래스를 예측하는 것이 불가능 한 일이였으나 CLIP은 이것이 가능하게 됨
	  이는 기존의 Label을 사용하여 이미지의 클래스를 구분하는 방식이 아닌, 이미지와 자연어의 정렬 (Align)을 학습했기 때문에
	> 사람과의 성능비교
	  Accuracy 성능만 놓고 비교해볼때, 저자들이 주목하는건 OneShot 성능
	  사람은 Zero Shot 성능은 크게 떨어지지만, 하나의 샘플을 참고하고 나면 크게 점수가 오름. 웃긴건 두 개의 샘플을 본다고 해서 성능이 더더욱 좋아지는 건 아님
	  반면 CLIP은 Zero Shot 성능은 좋지만, 하나의 샘플을 사용하여 재학습시 오히려 성능이 떨어짐
	  
	  이건 CLIP의 특성과도 연관이 있는데
	  Zero Shot 성능에 특화되도록 설계된 CLIP이 그동안의 학습 과정보다(혹은 학습과정을 무시하고) Few Shot으로 재학습하는 과정에서 기존 학습된 파라미터가 모두 변경되며 성능이 하락하는 것으로 추측
	  반면 인간의 경우에서는 단 하나의 샘플이 주어졌어도 기존의 지식과 비교해가며 새로운 정보를 일반화하고 해석하는데, 이를 메타인지 라고 한다
	  저자들은 CLIP에는 이런 부분이 부족함을 강조함
