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
>>![Alt text](/images/img_0_1.PNG)
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
>참고로 다크프로그래머님은 한국에서 영상처리 (블로그) 최고 교수님격이다.

___
몰랏던 표현들

- miscellaneous: 여러가지 잡다한, 다방면의
- deficiency: 부족, 결함
- nonintrusive: 개입하는
- holistic: 총체주의적
