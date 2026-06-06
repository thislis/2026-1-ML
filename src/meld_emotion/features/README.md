# features — 특징 추출기

`FeatureExtractor` 는 한 모달리티에서 한 종류(임베딩/개념)의 특징을 뽑는다. 학습이 필요한
추출기는 `fit` 에서 상태를 학습하고, 없으면 no-op 이다. 모달리티가 없는 샘플은 0 벡터를
반환한다(가용성은 파이프라인이 별도 마스크로 관리).

## 구성 (모달리티 × 종류)

| 모달리티 | 개념(concept, c_T/c_A/c_V) | 임베딩(embedding) |
| --- | --- | --- |
| text | `TextConceptExtractor` ✅ | `BowTextExtractor` ✅ · `EmbeddingGemmaTextExtractor` ✅ · `TfidfTextExtractor` ⚠️ · `SentenceEmbeddingExtractor` ⚠️ |
| audio | `AudioConceptExtractor` ✅ | `Wav2Vec2XlsrAudioExtractor` ✅ · `MfccAcousticExtractor` ⚠️ |
| video | `VideoConceptExtractor` ✅ | `VisualCueExtractor` ⚠️ |

✅ 완전 구현 · ⚠️ 임시(placeholder, 결정적 수치 특징 반환 + 경고). 개념 추출기는
제안서의 해석 가능한 개념 벡터 `c = [c_T, c_A, c_V]` 를 구성한다.

`EmbeddingGemmaTextExtractor` 는 `sentence-transformers` 로 `google/embeddinggemma-300m` 을
lazy-load 해 768차원 dense embedding 을 만든다. `output_dim` 을 128/256/512/768 로 지정하면
EmbeddingGemma 의 Matryoshka 표현을 앞 차원 truncate + 재정규화 방식으로 사용한다. 실행 전
`uv sync --extra text` 가 필요하고, Hugging Face 에서 Google Gemma 라이선스 조건에 동의해야
모델 파일에 접근할 수 있다.
`prompt_name` 은 모델이 제공하는 prompt key 와 정확히 일치해야 하므로 기본값은 대문자
`Classification` 이다.

최초 사용 전 [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)
모델 페이지에서 로그인 후 라이선스에 동의하고, Hugging Face Read token 으로 CLI 인증을 마친다.

```bash
uv run huggingface-cli login
uv run huggingface-cli whoami
```

CI/원격 실행처럼 대화형 로그인이 어렵다면 `HF_TOKEN=hf_...` 환경변수를 설정한다.

```yaml
extractors:
  - type: text_embeddinggemma
    model_name: google/embeddinggemma-300m
    output_dim: 768
    batch_size: 32
    normalize: true
    prompt_name: Classification
```

`Wav2Vec2XlsrAudioExtractor` 는 `transformers`/PyTorch 로
`facebook/wav2vec2-xls-r-300m` 을 lazy-load 해 마지막 hidden state 를 mask-aware mean pooling
으로 1024차원 발화 임베딩으로 바꾼다. 오디오가 없는 샘플은 0 벡터를 반환한다. 실행 전
`uv sync --extra audio` 가 필요하며, 입력 waveform 은 16kHz mono 여야 한다. 이 base checkpoint
는 tokenizer vocab 이 없으므로 `AutoProcessor` 가 아니라 `AutoFeatureExtractor` 로 로드한다.
모델 로딩 실패 시 의존성/모델 접근성/입력 sample rate 를 확인할 수 있도록 원래 예외 원인을
포함해 `RuntimeError` 를 낸다.
`chunk_seconds` 를 지정하면 긴 waveform 을 여러 chunk 로 나눠 각 chunk 임베딩을 계산한 뒤
길이 가중 평균해 self-attention 메모리 폭증을 피한다.

```yaml
extractors:
  - type: audio_wav2vec2_xlsr
    model_name: facebook/wav2vec2-xls-r-300m
    output_dim: 1024
    batch_size: 4
    sampling_rate: 16000
    chunk_seconds: 30.0
    normalize: true
```

MELD 팀이 제공한 baseline pickle 을 쓰는 `MeldPrecomputedFeatureExtractor` 도 완전 구현되어
있다. 이 추출기는 설정에서 `type: meld_precomputed`, `path`, `modality`, `kind` 를 받아
`FeatureMatrix` 로 변환하며, 데이터셋은 `MeldDatasetSource(metadata_path=...)` 와 함께 쓰는
것이 기본 경로다.

## 새 특징 추출기 추가하기

1. [base.py](base.py) 의 `BaseFeatureExtractor` 를 상속하고 `modality`, `kind`(`ClassVar`),
   `transform(samples) -> FeatureMatrix` 를 구현한다(상태 학습이 필요하면 `fit` 도). 상태 마커
   `@real`/`@placeholder` 를 붙인다.
2. [config/schema.py](../config/README.md) 에 `ExtractorConfig` 하위 설정 추가·등록.
3. [pipeline/builder.py](../pipeline/builder.py) `build_extractor` 에 분기 추가.
4. 새 extractor 를 YAML 에서 쓰려면 필요한 optional dependency 를 `pyproject.toml` extra 에도
   맞춰 둔다.

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
(scikit-learn/sentence-transformers/transformers/librosa/mediapipe 등)로 채울 때 `@placeholder` 를 떼고 `@real` 로 바꾸면
`meld-emotion status` 에 자동 반영된다. 무거운 의존성은 `[text]`/`[audio]`/`[video]` extra.
