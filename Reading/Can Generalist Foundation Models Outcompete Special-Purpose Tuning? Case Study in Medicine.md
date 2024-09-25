Can Generalist Foundation Models Outcompete Special-Purpose Tuning?
Case Study in Medicine
(범용모델이 특수한 목적으로 미세조정(학습)된 모델을 능가할 수 있을까?)
(의학 분야에서의 사례연구)

Microsoft
November 2023


Abstract
Generalist foundation models such as GPT-4 have displayed surprising capabilities in a wide variety of domains and tasks.
(GPT-4와 같은 범용 모델은 매우 다양한 영영과 작업에서 놀라운 능력을 보여주고 있다)
Yet, there is a prevalent assumption that they cannot match specialist capabilities without intensive training of models with specialty knowledge.
(그러나 전문지식으로 집중학습 시킨 모델이 아닌 이상 전문가 역량에 필적할 수 없다는 의견이 지배적)
For example, most explorations to date on medical competency benchmarks have leveraged domain specific training, as exemplified by efforts on BioGPT and Med-PaLM.
(예를들어, 의료 벤치마크에서의 탐색 대부분은 특정 도메인으로 학습된 BioGPT 및 Med-PaLM으로 이루어짐)

...

Our experimental design carefully controls for overfitting during the prompt engineering process.
(우리는 엔지니어링 과정에서 과적합을 신중하게 제어하며 실험 설계하였음)
As a culmination of the study, we introduce Medprompt, based on a composition of several prompting strategies.
(연구의 마지막으로, 몇가지 자극적인 전략으로 구성된 MedPrompt를 소개할 것이다)
Medprompt greatly enhances GPT-4’s performance and achieves state of the art results on all nine of the benchmark datasets in the MultiMedQA suite.
(Medprompt는 GPT-4의 성능을 크게 향상시키고 MultiMedQA 제품군의 9개 벤치마크 데이터 셋 모두에서 좋은 결과를 달성했다)
The method outperforms state-of the-art specialist models such as Med-PaLM 2 by a large margin with an order of magnitude fewer calls to the model.
(이 방법은 Med-PaLM 2와 같은 좋은 전문 모델을 능가하며 모델 호출 수가 훨씬 적다)
Steering GPT-4 with Medprompt achieves a 27% reduction in error rate on the MedQA dataset (USMLE exam) over the best methods to date achieved with specialist models, and surpasses a score of 90% for the first time.
(Medprompt를 사용한 GPT-4는 전문 모델을 통해 달성한-현재까지 가장 우수한 방법이라고 알려진 것에 비해- MedQA 데이터 셋(USMLE 검사)에서 오류율을 27% 감소시키면서도 90%의 점수를 넘었음)
Moving beyond medical challenge problems, we show the power of Medprompt to generalize to other domains and provide evidence for the broad applicability of the approach via studies of the strategy on competency exams in electrical engineering, machine learning, philosophy, accounting, law, nursing, and clinical psychology.
(의학문제 뿐 아니라 전기공학, 기계 학습, 철학, 회계, 법률, 간호 및 임상 심리학의 역량 검사 전략에 대한 연구를 통해 다른 영역으로 일반화하고 접근 방식의 광범위한 적용 가능성에 대한 증거를 제공하는 메드프롬의 힘을 보여줄 것)





1. Introduction
A long-term aspiration in AI research is to develop principles of computational intelligence and to harness these to build learning and reasoning systems that can perform general problem solving across a diversity of tasks [21, 22].
(AI 연구의 장기적 목표는 다양한 작업에 걸처 문제 해결을 수행할 수 있는 학습 및 추론 시스템을 구축하는 것)
In line with this goal, large language models, also referred to as foundation models, such as GPT-3 [3] and GPT-4 [24], have demonstrated surprising competencies on a broad swath of tasks without requiring heavy specialized training [4].
(이 목표에 따라 GPT-3, 4와 같은 (파운데이션 모델이라고도 하는) 대규모 언어모델은 광범위한 작업에서 놀라운 능력을 입증함)
These models build on the text-to-text paradigm [31] with investments in compute and data to learn at scale from indiscriminate consumption of large amounts of public web data.
(이런 모델은 대량의 공개된 웹 데이터를 텍스트간 관계를 기반으로 컴퓨팅 및 데이터에 대한 투자를 함)
Some of these models are tuned via a learning objective to perform general instruction-following via prompts
(이런 모델 중 일부는 학습 목표를 통해 조정되며, 프롬프트를 통해 일반적인 지침 따르기를 수행한다)

A core metric for characterizing the performance of foundation models is the accuracy of next word prediction.
(기초 모델의 성능을 특성화하기 위한 핵심 메트릭은 '다음 단어의 예측' 정확도)
Accuracy with next word prediction is found to increase with scale in training data, model parameters, and compute, in accordance with empirically derived “neural model scaling laws” [3, 12]).
('다음 단어를 예측'하는 것의 정확도는 경험적으로 도출된 '신경 모델 스케일링 법칙'에 따라 훈련 데이터, 모델 매개변수 및 계산에서 규모에 따라 증가하는 것으로 나타남)
However, beyond predictions of scaling laws on basic measures such as next word prediction, foundation models show the sudden emergence of numerous problem-solving capabilities at different thresholds of scale [33, 27, 24].
(그러나 다음 단어 예측과 같은 기본 능력에 대한 '스케일링 법칙'의 예측너머, 기반 모델은 다양한 규모의 임계값에서 수많은 문제 해결 능력의 갑작스러운 출현을 보여준다.)?
Despite the observed emergence of sets of general capabilities, questions remain about whether truly exceptional performance can be achieved on challenges within specialty areas like medicine in the absence of extensive specialized training or fine-tuning of the general models.
(일반적인 가능성이 나타났음에도 불구하고, 광범위한 전문교육이나 일반 모델의 미세조정이 없는 상태에서 '의학'과 같은 전문 부냥 내의 과제에서 진정으로 탁월한 성능을 달성할 수 있는지에 대해서는 아직 의문이 남음)
Most explorations of foundation model capability on biomedical applications rely heavily on domain- and task-specific fine-tuning.
(생물 의학 응용 분야에서 기초 모델 기능에 대한 대부분의 탐색은 도메인 및 작업별 미세조정에 크게 의존하는 상황)
With first-generation foundation models, the community found an unambiguous advantage with domain-specific pretraining, as exemplified by popular models in biomedicine such a PubMedBERT [10] and BioGPT [19].
(1세대 기반모델을 사용하여, PubMedBERT 및 BioGPT와 같은 도메인별 사전교육을 통한 명확한 이점을 발견했었음)
We focus in this paper on steering foundation models via prompt engineering to excel on a set of medical challenge benchmarks.
(본 논문에서는 일련의 '의료 과제 벤치마크'에서 prompt engineering을 통한 우수함에 초점을 맞추고 있음) 아마도..
Med-PaLM 2 attains competitive results on MedQA and other medical challenge problems, via expensive, task-specific fine-tuning of the general PaLM [6] foundation model [29, 30].
(Med-PaLM 2는 일반 PaLM에 값비싼 작업별 미세조정을 함으로 MedQA 및 기타 의료 과제 문제에 대해 경쟁력 있는 결과를 얻었음)
....
Shortly after GPT-4 was made public in March 2023, several co-authors of this study showed that the model had impressive biomedical competencies “out-of-the-box” on medical challenge benchmarks.
(2023년 3월 GPT-4가 공개된 직후, 이 연구의 여러 공동 저자는 이 모델이 의학적 도전 벤치마크에서 인상적인 생물의학 역량을 "즉각적으로" 가지고 있음을 보여주었다)
To demonstrate the latent power of GPT-4 on specialty medical expertise, the coauthors purposefully employed a rudimentary prompting strategy [23].
(전문 의료 전문 지식에 대한 GPT-4의 잠재력을 입증하기 위해 공동 저자는 의도적으로 기본적인 유도 전략을 사용)
Despite the strong results demonstrated in that study, questions remain about the depth of GPT-4’s domain-specific capabilities in the absence of additional special training or tuning.
(해당 연구에서 강력한 결과가 입증되었지만, 여전히 추가적인 훈련이나 튜닝이 없는 상태에서 '도메인별 기능'의 깊이에 대해서는 의문)
....
Medprompt unleashes medical specialist skills in GPT-4 in the absence of expert crafting, easily topping existing benchmarks for all standard medical question-answering datasets.
(Medprompt는 GPT-4가 전문가 제작이 없는 상태에서 모든 '표준 의료 질문-답변 데이터 셋'의 기존 벤치마크에서 쉽게 1위를 차지할 수 있게 함)
....
We discover that a combination of methods, including in-context learning and chain-of-thought, can yield synergistic effects.
(우리는 chain-of-thought를 포함한 다양한 방법들이 시너지 효과를 낼 수 있다는 사실을 발견)
Perhaps most interestingly, we find that the best strategy in steering a generalist model like GPT-4 to excel on the medical specialist workload that we study is to use a generalist prompt.
(흥미로운것은 GPT-4와 같은 범용 모델을 이용하여 의료 전문가의 작업량을 능가하는 최선의 방법은 '일반적인 프롬프트'를 사용하는 것이라는 걸 발견함)
We find that GPT-4 benefits significantly from being allowed to design its prompt, specifically with coming up with its own chain-of-thought to be used for in-context learning.
(우리는 GPT-4가 프롬프트를 설계하도록 허용함으로써, 특히 맥락 학습에 사용할 수 있는 자체적인 chain-of-thought를 고안하는 것이 상당한 이점이 된다는 걸 발견)
This observation echoes other reports that GPT-4 has an emergent self-improving capability via introspection, such as self-verification [9].
(이 관찰은 GPT-4가 자기 검증과 같은 성찰을 통해 새로운 자기 개선 능력을 가지고 있다는 다른 보고서를 반영한다)
We note that the automated chain-of-thought reasoning removes dependency on special human expertise and medical datasets.
(우리는 자동화된 chain-of-thought가 특수한 전문 지식이나 의료 데이터 셋에 대한 의존성을 제거한다는 점에 주목한다)
Thus, despite the name Medprompt, extending from the framing context and research trajectory of our investigation of the capabilities of GPT-4 on medical challenge problems, the methodology doesn’t include any components specifically oriented towards medicine.
(따라서, 의학적인 문제에 대한 GPT-4의 능력을 조사하는 맥락과 Medprompt라는 이름에도 불구하고, 제시하는 이 방법론에는 의학을 특별히 지향하는 구성요소가 포함되어 있지 않음)
As we explore in Section 5.3, the approach can be applied readily to other domains.
(5.3절에서 살펴볼 수 있을, 이 접근 방식은 다른 도메인에 대해서도 쉽게 적용될 것이라고 생각)





2. Background
2.1 Foundation Models on Medical challenge Problems
In the era of first-generation foundation models, limited model size and computational resources made domain-specific pretraining advantageous.
(1세대 'foundation models' 시대에는 제한된 모델 크기와 계산 리소스가 도메인별 사전 훈련을 유리하게 만들어줌)
....
More powerful, general-domain foundation models have demonstrated significantly elevated performance in medical challenges without requiring domain-specific pretraining.
(보다 강력한 범용(일반 도메인) 모델은 도메인별 사전 교육 없이도 의료 문제에서 향상된 성능을 입증)
....
Other studies have explored the power of relying on explicit tuning with medical knowledge.
(다른 연구들은 의학 지식을 가지고 '명시적인 튜닝'에 의존하는 것을 탐구)
....
We re-examine the capabilities of generalist foundation models without resorting to extensive fine-tuning.
(우리는 광범위한 미세조정에 의존하지 않고 일반적인(범용) 모델의 기능을 재검토)
We explore diverse prompting strategies to best steer powerful generalist foundation models toward delivering strong performance in specialized domains.
(우리는 전문 영역에서 강력한 성능을 제공하기 위해 가장 잘 조정하기 위한 다양한 유도 전략을 모색한다)



2.2 Prompting Stratergies
Prompting in the context of language models refers to the input given to a model to guide the output that it generates.
(언어 모델에서 prompt의 맥락은 모델이 생성하는 출력을 '안내하기 위해' 모델에 제공되는 입력을 의미한다.)
Empirical studies have shown that the performance of foundation models on a specific task can be heavily influenced by the prompt, often in surprising ways.
(경험적 연구에 따르면 특정 작업에 대한 범용 모델의 성능은 종종 놀랍도록 크게 prompt에 영향을 받는다)
For example, recent work shows that model performance on the GSM8K benchmark dataset can vary by over 10% without any changes to the model’s learned parameters [35].
(예를 들어, 최근 연구에 따르면 GSM8K 벤치마크 데이터 세트의 모델 성능은 모델의 학습된 매개 변수를 변경하지 않고도 10% 이상 차이가 날 수 있다)
Prompt engineering refers to the process of developing effective prompting techniques that enable foundation models to better solve specific tasks.
(프롬프트 엔지니어링은 기초 모델이 특정 작업을 더 잘 해결할 수 있도록 하는 효과적인 프롬프트 기법을 개발하는 과정을 말한다)
Here, we briefly introduce a few key concepts that serve as building blocks for our Medprompt approach.
(여기서는 Medprompt 접근 방식의 구성 요소가 되는 몇 가지 주요 개념을 간단히 소개한다)
In-Context Learning (ICL) is a key capability of foundation models, allowing the models to solve new tasks from just a few task demonstrations [3].
(In-Context Learning(ICL)은 기초 모델의 핵심 기능으로, 모델이 몇 번의 과제 시연만으로도 새로운 과제를 해결할 수 있게 한다)
For example, an ICL prompt can be created by preceding a test question with several different examples of questions and desired results.
(예를 들어, ICL 프롬프트는 여러가지 다른 '예시질문'과 '원하는 결과'를 포함하는 test 질문 앞에 작성할 수 있다)
ICL does not require updating model parameters but can offer effects similar to fine-tuning.
(ICL은 모델 파라미터를 업데이트할 필요 없이 미세 조정과 유사한 효과를 제공할 수 있다)
The choice of examples used in few-shot prompting can substantially influence model performance.
(few shot prompt에 사용되는 예제의 선택은 모델 성능에 상당한 영향을 미칠 수 있다)
....
Chain of Thought (CoT) is a prompting methodology that employs intermediate reasoning steps prior to introducing the sample answer [34].
(chain-of-thought(CoT)는 답변을 출력하기 전에 중간 추론 단계를 사용하는 프롬프트 방법론)
By breaking down complex problems into a series of smaller steps, CoT is thought to help a foundation model to generate a more accurate answer.
(CoT는 복잡한 문제를 일련의 작은 단계로 분해함으로써 기초 모델이 보다 정확한 답을 생성하는 데 도움이 될 것으로 생각)
CoT ICL prompting integrates the intermediate reasoning steps of CoT directly into the few-shot demonstrations.
(CoT ICL prompting 은 CoT 의 중간 추론 단계를 few-shot 시연에 직접 통합합니다)?
....
As we shall describe in more detail, we can do this successfully by providing [question, correct answer] pairs from a training dataset.
(더 자세히 설명하겠지만, 우리는 교육 데이터 세트에서 [질문, 정답] 쌍을 제공함으로써 이를 성공적으로 수행할 수 있다)
We find that GPT-4 is capable of autonomously generating high-quality, detailed CoT prompts, even for the most complex medical challenges.
(우리는 GPT-4가 가장 복잡한 의학적 과제에도 고품질의 상세한 CoT 프롬프트를 자율적으로 생성할 수 있다는 것을 발견했다)
Ensembling is a technique for combining the outputs of multiple model runs to arrive at a more robust or accurate result via combining the separate outputs with functions like averaging, consensus, or majority vote.
(앙상블은 여러 모델 실행의 출력을 평균화, 합의 또는 다수결과 같은 기능과 결합하여 보다 강력하거나 정확한 결과를 얻기 위한 방법)
....
The diversity of the outputs can be controlled by shifting the “temperature” parameter in a model’s generation, where higher temperatures can be viewed as injecting greater amounts of randomness into the generation process.
(출력의 다양성은 모델의 세대에서 "temperature" 매개변수를 이동하여 제어할 수 있으며, temperature가 높으면 생성 프로세스에 더 많은 양의 무작위성이 주입되는 것으로 볼 수 있다)
By reordering or shuffling components of a few-shot prompt, ensembling techniques can also address the order sensitivity commonly found with foundation models [26, 39], thus improving robustness.
(few-shot prompt의 구성 요소를 순서 변경하거나 셔플링함으로써 앙상블 기법은 기초 모델에서 흔히 발견되는 순서 민감도를 해결할 수 있으므로 견고성이 향상)
While ensembling can enhance performance, it comes at the cost of increased computational demands.
(앙상블은 성능을 향상시킬 수 있지만, 증가된 계산 요구에 따른 비용이 발생합니다)
For example, Med-PaLM 2’s Ensemble Refinement method used as many as 44 separate inferences for a single question.
(예를 들어, Med-PaLM 2의 앙상블 Refinement는 하나의 질문에 대해 44개의 개별 추론을 사용)
....





3 Experimental Design
We start with an overview of the medical challenge problem datasets and then outline our testing methodology, designed to avoid overfitting that can occur with intensive iteration on a fixed evaluation dataset.
(의료 과제 문제 데이터 세트에 대한 개요로 시작한 다음 고정 평가 데이터 세트에서 집중적인 반복으로 발생할 수 있는 과적합을 방지하도록 설계된 테스트 방법론을 설명할 것)
3.1 Datasets
Specifically, the benchmarks include the following:
(구체적인 벤치마크는 아래와 같다)
(MedQA, MedMCQA, PubMedQA, MMLU 및 그에대한 상세내용)
As we shall see in Section 5.3, we can test the generality of the Medprompt approach by studying its efficacy for competency exams outside the primary focus on medical challenge problems.
(5.3절에서 살펴보겠지만, 우리는 의료 문제에 대한 일차적인 초점 이외의 역량 시험에 대한 효과를 연구함으로써 메드프롬프트 접근법의 일반성을 테스트할 수 있다)
We test our methodology on two nursing datasets focused on answering NCLEX (National Council Licensure Examinaton) questions and six additional datasets from MMLU covering topics like accounting and law.
(NCLEX(National Council License Examinaton) 질문에 답변하는 데 중점을 둔 2개의 간호 데이터 세트와 회계 및 법률과 같은 주제를 다루는 MMLU의 6개의 추가 데이터 세트에 대해 방법론을 테스트)
Details of these datasets are presented in Section 5.3.
(데이터 세트에 대한 자세한 내용은 섹션 5.3)


3.2 Sound Testing methodology
While prompting and in-context learning does not change model parameters, a specific choice of prompting strategy can be viewed as a high-level setting or hyperparameter of the end-to-end testing process.
(prompt 나 in-context는 모델 매개변수를 변경하지 않지만, end-to-end 테스트 process의 높은 수준의 설정 또는 hyperparameter 로 볼 수는 있다.)
As a result, we must be cautious about overfitting as part of training and testing, thus providing results that would not generalize out of the training and test sets under consideration.
(따라서 훈련 및 테스트의 일환으로 과적합에 주의해야 하며, 고려 중인 훈련 및 테스트 세트에서 일반화되지 않는 결과를 제공해야한다)
Concerns about overfitting with studies of foundation model performance are similar to the valid concerns in traditional machine learning with overfitting during the hyperparameter optimization process [8].
(기초 모델 성능에 대한 연구로 과적합에 대한 우려는 하이퍼파라미터 최적화 프로세스 동안 과적합으로 전통적인 기계 학습에서 유효한 우려와 유사)
We wish to avoid analogous overfitting in the prompt engineering process.
(우리는 overfitting과 유사한 prompt engineering은 피하기를 원한다)
Intuitively, a prompt harnessing for examples a lookup table of specific benchmark questions will naturally perform much better on those questions than on unseen problems.
(직관적으로, 특정 벤치마크 '질문'을 예로 들어 활용하는 것은 실제 질문에 대해 훨씬 더 자연스럽게 잘 수행될 것이기에)
A common technique to address this problem in traditional machine learning is to create “test” sets, which are only evaluated against at the end of the model selection process.
(전통적인 기계학습에서 이 문제를 해결하기 위한 일반적인 기술은 '테스트'셋을 만드는 것이며, 이 셋은 모델 선택 프로세스가 끝날 때만 비교 평가 된다)
We adopt this important aspect of sound testing methodology for machine learning studies and randomly carved out 20% of each benchmark dataset as an “eyes-off” split that is completely held out from consideration until the final testing phase.
(우리는 각 벤치마크의 데이터 셋의 20%를 최종 테스트 단계까지 완전히 고려하지 않는 'eyes-off' 분할한다.)
That is, the eyes-off data is kept hidden until the end-stage.
(이렇게 분할된 데이터는 종료 단계까지 숨겨진다)
The data is not examined or optimized against during the prompt engineering process.
(prompt engineering process 중에는 이 데이터가 검사되거나 최적화에 사용되지 않는다)
....
We find that our performance is quite similar between the two, and that GPT-4 with Medprompt actually performs marginally better on the eyes off, held out data suggesting that the methods will generalize well to similar questions in the “open world.”
(“eyes-on” vs. “eyes-off” 두가지의 성능이 상당히 비슷하며, eyes-off가 약간 더 나은 성능을 발휘한다는 것을 발견)
We have not seen evidence of the use of a similar eyes-off approach in prior studies.
(우리는 이전 연구에서 "eyes-off"와 유사한 접근을 본적이 없다)





4 Power of Prompting: Exploration and Results
In this section, we detail the three major techniques employed in Medprompt: Dynamic few-shot selection, self-generated chain of thought, and choice shuffle ensembling.
(이 섹션에서는 메드프롬프트에 사용되는 세 가지 주요 기술인 동적 퓨샷 선택, 자체 생성 사고 체인 및 선택 셔플 앙상블을 자세히 설명할 것이다)
After discussing each technique, we review our approach to composing the three methods into the integrated Medprompt.
(각각의 기술에 대해 말한 후, 세 가지 방법을 통합하여 med-prompt로 구성하는 우리의 접근 방식을 검토할 것)

4.1 Dynamic Few-shot
