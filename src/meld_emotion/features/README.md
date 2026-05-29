# features — 특징 추출기

`FeatureExtractor` 는 한 모달리티에서 한 종류(임베딩/개념)의 특징을 뽑는다. 학습이 필요한
추출기는 `fit` 에서 상태를 학습하고, 없으면 no-op 이다. 모달리티가 없는 샘플은 0 벡터를
반환한다(가용성은 파이프라인이 별도 마스크로 관리).

## 구성 (모달리티 × 종류)

| 모달리티 | 개념(concept, c_T/c_A/c_V) | 임베딩(embedding) |
| --- | --- | --- |
| text | `TextConceptExtractor` ✅ | `BowTextExtractor` ✅ · `TfidfTextExtractor` ⚠️ · `SentenceEmbeddingExtractor` ⚠️ |
| audio | `AudioConceptExtractor` ✅ | `MfccAcousticExtractor` ⚠️ |
| video | `VideoConceptExtractor` ✅ | `VisualCueExtractor` ⚠️ |

✅ 완전 구현(numpy 전용) · ⚠️ 임시(placeholder, 결정적 수치 특징 반환 + 경고). 개념 추출기는
제안서의 해석 가능한 개념 벡터 `c = [c_T, c_A, c_V]` 를 구성한다.

## 새 특징 추출기 추가하기

1. [base.py](base.py) 의 `BaseFeatureExtractor` 를 상속하고 `modality`, `kind`(`ClassVar`),
   `transform(samples) -> FeatureMatrix` 를 구현한다(상태 학습이 필요하면 `fit` 도). 상태 마커
   `@real`/`@placeholder` 를 붙인다.
2. [config/schema.py](../config/README.md) 에 `ExtractorConfig` 하위 설정 추가·등록.
3. [pipeline/builder.py](../pipeline/builder.py) `build_extractor` 에 분기 추가.

```python
@real
class MyTextExtractor(BaseFeatureExtractor):
    modality: ClassVar[Modality] = Modality.TEXT
    kind: ClassVar[FeatureKind] = FeatureKind.EMBEDDING
    def transform(self, samples):
        rows = [self._vec(s.text) for s in samples]
        return self._stack_rows(rows, self.names)
```

## 임시(placeholder) 교체

`features/text/tfidf.py` 등은 결정적 대체 특징을 반환하며 사용 시 한 번 경고한다. 실제 라이브러리
(scikit-learn/librosa/opencv)로 채울 때 `@placeholder` 를 떼고 `@real` 로 바꾸면
`meld-emotion status` 에 자동 반영된다. 무거운 의존성은 `[text]`/`[audio]`/`[video]` extra.
