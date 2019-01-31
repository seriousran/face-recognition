# Deep Face Recognition A survey 논문 리뷰

```BibTex
@article{Wang2018DeepFR,
  title={Deep Face Recognition: A Survey},
  author={Mei Wang and Weihong Deng},
  journal={CoRR},
  year={2018},
  volume={abs/1804.06655}
}
```

### Abstract | 초록
딥러닝은 다중 처리 레이어(multiple layers)를 적용하여 다중 레벨 특징 추출(multiple levels of feature extraction)을 사용한 데이터 표현을 학습합니다.
이 떠오르는 기술은 Deepface와 DeepID를 통해 얼굴 인식(Face Recognition) 연구를 재구성하였습니다. 그 이후로 얼굴 인식에서 엄청난 성능을 보입니다!
본 논문에서는 알고리즘, 데이터, 장면에 대한 광범위한 주제를 다루는 딥러닝을 이용한 얼굴인식에서 최근에 이루어진 진언에 대한 포괄적인 개관(survey)을 제공합니다.
1. 다양한 네트워크 아키텍쳐와 손실 함수(loss functions)들을 요악하고,
2. 두 개의 분류로 나눠지는 관련된 얼굴 처리 방법들,
    - one-to-many augmentation (일대다 증가/증강)
    - many-to-one normalization (다대일 표준화)
3. 딥러닝을 이용한 얼굴인식 분야에서의 여러가지 잡다한 리뷰
    - cross-factor, heterogenous, multiple-media, and industry scenes
4. 현재 방법들의 잠재적인 결함과 몇몇 조명받을 미래 방향들

## 1. Introduction
여기저기 많이 쓰인다는 이야기...<br/>
**1990년대 초**에 historical <u>Eigenface</u>(고유 얼굴) approach가 소개되면서 얼굴 인식 연구는 대중적이 되었습니다[164].<br/>
>~~대학원 입학했을 때, PCA와 함께 열심히 과제로 고생하던 기억이...~~<br/>
feature-based(특징점 기반) 얼굴인식의 마일스톤은 그림.1에 잘 표현되어있습니다.

![Alt text](/images/img_0_0.PNG)

>~~그림이 작아서 잘 안보일 수도 있으니 주의. 반 자르려 했으나 자료가 수평적인 연결성이 있어서....(핑계)~~<br/>
성능평가는 LFW dataset을 기준으로 Egienface(60%), Gabor/LBP(70%), LE(82%), 그리고 Deepface(97%) 순으로 성능이 향상되는 것을 보여줍니다.
>현재는, 99.에서 소수점 단위로 향상되는, 혹은 시각화된 정보로서 잘 클러스터 된 것을 보여주는 방식으로 발전하는 중입니다.

**1990년대와 2000년대 사이** 총체주의적 접근방법(<u>holistic approaches</u>)은 특정한 분포 가정을 통한 low-dimensional 표현을 이끌어냈습니다. 여기서 low-dimensional 표현은 linear subspace나 manifold, 그리고 sparse representation과 같은 것을 말합니다.
>1. linear subspace[13][118][44]는 [성윤님의 블로그](https://hwauni.tistory.com/entry/Linear-subspaces)를 참고<br>
>2. manifold[70][199][43]: 다양체 학습 <br>
>> ![Alt text](/images/img_0_1.png) <br>
>>똑같지만 독특한 얼굴들의 다양성을 나타내는 사진입니다. 얼굴은 무작위로 manifold로 샘플링됩니다. 매개변수 벡터를 선형으로 스케일링(늘리고/줄이고)하면, distinctiveness(고유성/차별성)는 달라지지만, ID(그 사람이 누구인지)는 여전히 고정됩니다. 샘플을 manifold상에서 이동시키면 distinctiveness가 고정된 상태에서 ID가 달라집니다.<br>
>>ref: [Manifold-based constraints for operations in face space](https://www.semanticscholar.org/paper/Manifold-based-constraints-for-operations-in-face-Patel-Smith/531ed96d7eb85397123e9ab96e0866a986a96ff5)<br>
>>about distintiveness: [What's distinctive about a distinctive face?](https://www.ncbi.nlm.nih.gov/pubmed/8177958)<br>
>3. sparse representation[184][221][40][42]: sparse한 표현 덕분에 전체의 일부분만으로도 얼굴을 인식할 수 있음.(ex:가려진 얼굴에 성능 좋음)<br>

**2000년대 초**에 통제되지않은 얼굴의 변화 문제는 <u>local feature-based</u> 얼굴인식을 통해 해결되었습니다.
>사실 이건 얼굴 인식 뿐만 아니라, 모든 영상처리 분야에서 혁명을 불러왔죠.
<u>Gabor</u>와 <u>LBP</u>는 multi-level과 high-dimensional extensions 뿐만 아니라, local filterning의 몇 불변 속성을 통해 견고한 성능을 달성했습니다. 하지만, 불행히도 직접 서술한 특징들은 distinctiveness와 compactness의 부족으로 어려움을 겪었습니다.

**2010년대 초**, <u>학습 기반의 local descriptor</u>(기술자/서술자)가 얼굴인식 커뮤니티에 도입되었는데, local filter가 더 나은 distinctiveness를 위해 학습되고, encoding codebook은 보다 더 compact한 것으로 학습됩니다.
>local feature descriptor는 [다크프로그래머 - 영상 특징점 추출방법의 7](https://darkpgmr.tistory.com/131)을 참고<br>
>encoding codebook은 [다크프로그래머 - Bag of Words 기법](https://darkpgmr.tistory.com/125)을 참고<br>
>참고로 다크프로그래머님의 블로그를 모른다면, 영상처리를 공부한 적이 없는 것이나 마찬가지.

얼굴 인식 문제를 해결하기 위해서 시도되었던 예전의 방법들은 일반적으로 한개 혹은 두개의 layer representation을 사용하였습니다. 예를들어,  filtering responses나 hostogram of the feature codes가 그러한 것들입니다. 연구자들은 preprocessing, local descriptors, 그리고 feature transformation을 개별적으로 향상시켜 얼굴인식의 정확도를 천천히 끌어올렸습니다. 계속적으로 발전해왔지만, 이러한 "shallow" methods는 LFW 벤치마크 성능을 95%까지밖에 향상시키지 못했습니다. 그래서 이러한 결과는 "shallow" methods가 안정적인 identity feature를 추줄하기에 불충분하다는 것을 나타냅니다. 그런데, **2012년**에 모든 것이 변했습니다. AlexNet이 ImageNet에서 우승을하며, 딥러닝이라는 methods가 존재감을 나타냅니다. 딥러닝은 feature extraction and transformation을 위해 cascade of multiple layers of processing units를 사용합니다. 딥러닝은 다른 수준의 추상(abstraction)에 대응하는 multple levels of representations를 학습합니다. 이러한 개념의 계층구조로부터 오는 levels는 얼굴 pose, 조명, 감정 변화에 강한 불변성을 띕니다. (아래의 그림 참고)

![Alt text](/images/img_0_2.png)

위 그림에서 보았을 때, 깊은 인공신경망에서 왼쪽에서부터 가장 첫 레이어는 Gabor feature와 비슷하게 나타납니다. 이전에 과학자들이 수년에 걸친 실험을 통해 찾아낸 것이었죠. 두번째 레이어는 좀 더 복잡한 texture features를 학습합니다. 세번째 레이어의 features는 더 복잡하고, 구조로서는 좀 단순한 구조들이 나타나기 시작합니다. 예를들어 high-bridged nose나 big eyes 같이요. 마지막 네번째 레이어에서는, 신경망의 출력이 특정한 얼굴의 속성을 충분히 나타내고 있습니다. 예를들어, 웃음, 소리 짖음, 심지어 파란 눈까지도요. 

**2014년에** DeepFace가 LFW 벤치마크의 state-of-the-art 정확도를 달성합니다. human performance에 접근하는 정도로 말이죠. `DeepFace: 97.35% vs. Human:97.53%` 이때 이후로, 얼굴 인식분야의 연구는 딥러닝 기반의 접근방법으로 모두 옮겨집니다. 그리고 그 정확도는 99.70%까지 드라마틱하게 올라갑니다. 단지 3년만에요. 딥러닝 기술은 얼굴 인식 연구분야를 완전히 뒤바꿔놉니다. 알고리즘, 데이터셋, 심지어 평가 프로토콜까도요.

### 논문에서 말하는 이 논문의 주요 contributions
1. 딥러닝을 이용한 얼굴인식에 대한 네트워크 아키텍처 및 손실 함수들의 발전에 대한 체계적인(분류법의) 리뷰
    - 다양한 손실 함수들을 유클리드거리 기반 손실, 코사인 마진 기반 손실, 소프트 맥스 손실 및 그 변형에 대해 연구
    - Deepface, DeepID 시리즈, VGGFace, FaceNet, VGGFafce 2, 그리고 다른 주류 네트워크 아키텍처
2. 포즈 변화에 따른 인식 난이도를 다루는 것과 같이 우리는 얼굴 처리 방법을 "one-to-many augmentation"과 "many-to-one normalization"의 두 클래스로 분류하고, 어떻게 GAN이 연구 분야를 촉진하는 지를 논의합니다.
3. 얼굴인식에 매우 중요한 활용 가능한 공공의 대규모 학습 데이터셋에 대한 비교 및 분석
    - LFW, IJB-A/B/C, Megaface, MS-Celeb-1M
    - 위 데이터셋들은 4가지 측면에서 리뷰하고 비교되었습니다.
        1. 학습 방법론
        2. 평가 작업
        3. 측정 항목
        4. 인식 장면
4. 일반적인 작업 외의 12개의 challenging 얼굴인식 scenes
    - ex) antispoofing, cross-pose 얼굴 인식, cross-age 얼굴 인식
    - 이러한 미해결 문제에 대해 특별히 고안된 방법을 검토함으로써 향후 얼굴인식에 대한 연구에서 중ㅇ요한 issue들을 밝힙니다.

### 이 논문의 구성
- Section 2: 몇가지 배경지식과 용어, 각 구성 요소 소개
- Section 3: 다른 네트워크 아키텍쳐와 손실함수
- Section 4: 얼굴을 처리하는 알고리즘과 데이터셋 요약
- Section 5: 다른 측면에서의 딥러닝을 이용한 몇가지 얼굴인식 방법을 간략하게 소개


## 2. Overview
### A. Background Concepts and Terminology
#### [130]에서 언급된 대로, 얼굴인식의 전체 시스템에 필요한 모듈은 아래 그림과 같이 3개가 있습니다.

![Alt text](/images/img_0_3.png)

1. Face Detection (includes Localization)
먼저 face detector를 사용하여 이미지 혹은 비디오의 얼굴을 localize합니다.
> 여기서의 localize는 어디인지 위치를 찾는 것입니다. 보통 이러한 위치는, 사각형으로 표현되며 시작점(x, y), 끝점(x, y)이나, 시작점(x, y), 크기(w, h)으로 표현됩니다. 픽셀 데이터에 접근할 때에는, 이것은 row인지 col인지, 대응하는 x, y값을 잘 설정하였는지 꼭 확인하여야 합니다. 또한, 이러한 사각 영역을 관심 영역이라고도 부르며, 영어로는 Region Of Interest(ROI)라고 합니다.
2. Facial Landmark Detection & Align
얼굴의 랜드마크(눈, 코, 입과 같은 것)을 검출하고, 정규화된(normalized) 표준 좌표로 정렬합니다.
> Facial Landmark: 입, 오른쪽 눈썹, 왼쪽 눈썹, 오른쪽 눈, 왼쪽 눈, 코, 턱이 가장 일반적인 landmark입니다. 하지만 이것을 표현하기 위한 방법은 다양합니다. 일반적으로는 dlib가 제공하는 68개의 점으로 이루어진 landmark를 많이 사용합니다. <br>
> ![Alt text](/images/img_0_4.png) <br>
> Align Face: 얼굴을 정렬하는 방법은 다양한 방법이 존재합니다. 가장 간단하게는 양쪽 눈과, 코, 입의 양 끝을 기준으로 5개의 위치를 지정해두고 모든 얼굴을 그에 맞춥니다. 그러면, 지정된 위치에 그 5개가 존재하기 때문에, 그것을 통해 학습을 하면 비교적 동일한 조건에서 학습 데이터를 가지고 학습을 할 수 있기 때문입니다. 위치적인 align 뿐만 아니라, 값도 normalize 하기 위해 평균 값을 빼거나 하는 방법을 사용하기도 합니다. 정확한 face alignment는 2D의 정보만으로는 어렵지만, 얼굴 인식을 위한 pre-processing으로 사용할 때에는 일반적으로 엄청나게 정교한 과정으로 align하지는 않습니다.
3. 얼굴 인식 모듈은 이러한 정렬된(aligned) 이미지로 구현됩니다.

#### 얼굴 인식의 분류
- Face Identification(얼굴 인식): 얼굴 인식은 일대 다 유사성을 계산하여 관찰한 얼굴의 특정 신원을 결정합니다.
- Face Verification(얼굴 검증): 얼굴 검증(Face Verification)은 갤러리와 관찰한 얼굴 사이의 일대일 유사도를 계산하여, 두 이미지가 같은 대상인지 여부를 확인합니다.

두 시나리오 모두 알려진 subjects 셋은 처음에는 시스템(갤러리)에 등록되고 테스트하는 동안 새로운 subject(probe)가 제공됩니다.

- Closed-set Identification: 관측하는 얼굴이 갤러리에 있는 어떤 id로 나타나는 경우
- Open-set Identification: 관측하는 얼굴이 갤러리에 없는 사람인 경우

___
몰랏던 표현들

- miscellaneous: 여러가지 잡다한, 다방면의
- deficiency: 부족, 결함
- nonintrusive: 개입하는
- holistic: 총체주의적
