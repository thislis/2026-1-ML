# features — 특징 추출기

`FeatureExtractor` 는 한 모달리티에서 한 종류(임베딩/개념)의 특징을 뽑는다. 학습이 필요한
추출기는 `fit` 에서 상태를 학습하고, 없으면 no-op 이다. 모달리티가 없는 샘플은 0 벡터를
반환한다(가용성은 파이프라인이 별도 마스크로 관리).

## 구성 (모달리티 × 종류)

| 모달리티 | 개념(concept, c_T/c_A/c_V) | 임베딩(embedding) |
| --- | --- | --- |
| text | `TextConceptExtractor` ✅ | `BowTextExtractor` ✅ · `EmbeddingGemmaTextExtractor` ✅ · `TextTokenEmbeddingExtractor` ✅ · `TfidfTextExtractor` ⚠️ · `SentenceEmbeddingExtractor` ⚠️ |
| audio | `AudioConceptExtractor` ✅ | `Wav2Vec2XlsrAudioExtractor` ✅ · `Wav2Vec2XlsrAudioSequenceExtractor` ✅ · `MfccAcousticExtractor` ⚠️ |
| video | `VideoConceptExtractor` ✅ | `TimeSformerVideoExtractor` ✅ · `VideoPrismVideoExtractor` ✅ · `VideoFrameEmbeddingExtractor` ✅ · `VisualCueExtractor` ⚠️ |
| multimodal | - | `JinaOmniMultimodalExtractor` ✅ |

✅ 완전 구현 · ⚠️ 임시(placeholder, 결정적 수치 특징 반환 + 경고). 개념 추출기는
제안서의 해석 가능한 개념 벡터 `c = [c_T, c_A, c_V]` 를 구성한다.
현재 `meld-emotion status` 기준 feature extractor 는 REAL 12개, PLACEHOLDER 4개다. placeholder 는
`TfidfTextExtractor`, `SentenceEmbeddingExtractor`, `MfccAcousticExtractor`, `VisualCueExtractor`
이며, 나머지 표의 extractor 는 실제 구현이다.

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

`JinaOmniMultimodalExtractor` 는 `jinaai/jina-embeddings-v5-omni-small` 에 텍스트, 오디오,
비디오를 한 번에 넣어 fused multimodal embedding 을 만든다. 출력은 `Modality.MULTIMODAL`
하나의 1024차원 `FeatureMatrix` 이며, `output_dim` 은 32/64/128/256/512/768/1024 로 줄일 수
있다. `device` 는 `cpu`, `mps`, `gpu` 중 하나이고 `gpu` 는 내부에서 `cuda` 로 매핑된다.
`max_video_frames` 는 Jina 에 넘기는 프레임 수를 제한하는 안전장치이며, MPS 에서는 8 이하를
권장한다.
실행 전 `uv sync --extra text --extra audio --extra video --extra deep` 가 필요하다.

```yaml
extractors:
  - type: jina_omni_multimodal
    model_name: jinaai/jina-embeddings-v5-omni-small
    output_dim: 1024
    task: classification
    device: cpu
    max_video_frames: 8
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

`TimeSformerVideoExtractor` 는 Facebook Research 의 TimeSformer 를 Hugging Face Transformers
`TimesformerModel` 로 lazy-load 해 `facebook/timesformer-base-finetuned-k400` checkpoint 에서
768차원 발화 임베딩을 만든다. 공식 TimeSformer 기본 모델과 맞춰 입력 프레임은 기본 8프레임,
224×224 RGB 로 균등 샘플링/resize 되고 ImageNet mean/std 로 정규화된다. 출력은 기본적으로
마지막 hidden state 의 CLS token 을 사용하며, `pooling: mean` 으로 token 평균 pooling 도 선택할
수 있다. 실행 전 `uv sync --extra video` 가 필요하다.

```yaml
extractors:
  - type: video_timesformer
    model_name: facebook/timesformer-base-finetuned-k400
    output_dim: 768
    batch_size: 2
    num_frames: 8
    frame_size: 224
    pooling: cls
    normalize: true
```

`VideoPrismVideoExtractor` 는 Google DeepMind VideoPrism JAX/Flax 구현으로
`google/videoprism-base-f16r288` 을 lazy-load 해 비디오 프레임을 768차원 발화 임베딩으로
바꾼다. 입력 프레임은 `(T,H,W,C)` RGB 또는 RGBA/gray 배열이면 되고, extractor 가 기본
16프레임을 균등 샘플링해 288×288 RGB, `[0,1]` 범위로 전처리한다. VideoPrism 의 출력 patch
token 은 모든 token 평균 풀링으로 하나의 벡터가 되며, `output_dim` 을 줄이면 앞 차원을
truncate 한 뒤 선택적으로 재정규화한다. 실행 전 `uv sync --extra video` 가 필요하다.
upstream `videoprism` 패키지는 `tensorflow-cpu` 를 의존성으로 선언하지만 macOS arm64 에서 해당
wheel 이 없어 설치가 실패할 수 있다. 프로젝트의 `pyproject.toml` 은 uv dependency metadata
override 로 `tensorflow-cpu` 를 lock/install 대상에서 제외하고, extractor 는 VideoPrism import
시에 필요한 `tensorflow.io.gfile` 만 경량 shim 으로 제공한다. 이 shim 은 Hugging Face 에서 받은
로컬 `.npz` checkpoint 를 읽는 VideoPrism base encoder 경로를 위한 것이다.

```yaml
extractors:
  - type: video_videoprism
    model_name: google/videoprism-base-f16r288
    output_dim: 768
    num_frames: 16
    frame_size: 288
    normalize: true
    prefer_batched_input: true
```

MELD 팀이 제공한 baseline pickle 을 쓰는 `MeldPrecomputedFeatureExtractor` 도 완전 구현되어
있다. 이 추출기는 설정에서 `type: meld_precomputed`, `path`, `modality`, `kind` 를 받아
`FeatureMatrix` 로 변환하며, 데이터셋은 `MeldDatasetSource(metadata_path=...)` 와 함께 쓰는
것이 기본 경로다.

세 foundation embedding 을 함께 쓰는 raw MELD suite 에서는 `text_embeddinggemma`,
`audio_wav2vec2_xlsr`, `video_timesformer` 를 조합한다. `FeaturePipeline` 은 raw media 를 chunk
단위로 준비해 각 extractor 에 전달하고, 동일 feature signature 의 suite 실험끼리는 in-memory
feature cache 를 공유한다.

## Fine-grained sequence extractor

단어/token, 오디오 구간, 비디오 프레임 단위 XAI 는 pooled utterance embedding 이 아니라
`SequenceFeatureMatrix(values=[n,L,D])` 를 내는 추출기를 사용한다. v1 구현은 다음 세 가지다.

- `TextTokenEmbeddingExtractor`(`type: text_token_embeddings`) — Hugging Face tokenizer/model 로
  token embedding 과 character span 을 보존한다.
- `Wav2Vec2XlsrAudioSequenceExtractor`(`type: audio_wav2vec2_xlsr_sequence`) — Wav2Vec2
  `last_hidden_state` 를 시간 step별 embedding 으로 보존한다.
- `VideoFrameEmbeddingExtractor`(`type: video_frame_embeddings`) — sampled frame 마다 CLIP vision
  embedding 을 계산해 frame-level importance 를 볼 수 있게 한다.

이 추출기들은 호환을 위해 pooled `FeatureMatrix` 도 함께 반환한다. 기존 early/late fusion 은
pooled matrix 를 쓰고, `dialogue_rnn` 은 sequence matrix 가 있으면 `[B,N,L,D]` 입력을 사용한다.

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
(scikit-learn TfidfVectorizer, sentence-transformers, librosa, mediapipe/OpenCV 등)로 채울 때
`@placeholder` 를 떼고 `@real` 로 바꾸면 `meld-emotion status` 에 자동 반영된다. 무거운 의존성은
`[text]`/`[audio]`/`[video]` extra 에 둔다.
