# Version History

## ours_v1

`ours_v1`은 MELD Raw의 CSV/MP4를 직접 전처리해서 특징을 뽑는 버전이 아니라, MELD metadata와 MELD 팀 제공 precomputed feature pickle을 이 프로젝트의 모듈형 파이프라인에 연결해 baseline 모델들과 dialogue-level PyTorch 모델을 비교한 버전이다.

실행 설정은 다음 파일 하나가 기준이다.

```bash
/Users/safeailab_macmini/Desktop/2026-1-ML/configs/meld_raw_train_test_suite_w_precomputed.yaml
```

재현 명령은 프로젝트 루트에서 실행한다.

```bash
cd /Users/safeailab_macmini/Desktop/2026-1-ML
uv sync --extra text --extra deep
uv run meld-emotion compare --config configs/meld_raw_train_test_suite_w_precomputed.yaml
```

`text` extra는 scikit-learn baseline에 필요하고, `deep` extra는 `dialogue_rnn`의 PyTorch 학습에 필요하다. 설정상 `dialogue_rnn.training.device`는 `mps`이므로 Apple Silicon MPS가 없거나 불안정하면 YAML에서 `device: cpu`로 바꿔야 한다.

### 사용 데이터와 경로

프로젝트 루트는 다음이다.

```text
/Users/safeailab_macmini/Desktop/2026-1-ML
```

`ours_v1`에서 실제 데이터 로딩 기준으로 쓰는 파일은 다음 세 개다.

```text
MELD.Features.Models/features/data_emotion.p
MELD.Features.Models/features/text_glove_average_emotion.pkl
MELD.Features.Models/features/audio_embeddings_feature_selection_emotion.pkl
```

`data_emotion.p`는 `MeldDatasetSource(metadata_path=...)`가 읽는 metadata pickle이다. YAML에는 `dataset.type: meld`만 지정되어 있고 `root`는 생략되어 기본값 `data/MELD`가 들어가지만, `metadata_path`가 설정되어 있기 때문에 CSV 로딩 경로는 사용되지 않는다. 즉 `MELD.Raw/train/train_sent_emo.csv`, `MELD.Raw/test_sent_emo.csv`, raw MP4 파일들은 이 suite 실행에서 직접 읽히지 않는다.

metadata split 매핑은 코드상 다음과 같다.

```text
Split.TRAIN -> "train"
Split.DEV   -> "val"
Split.TEST  -> "test"
```

`ours_v1` 설정은 `train_split: train`, `eval_split: test`이므로 train과 test만 학습/평가에 사용한다. 확인된 metadata 규모는 다음과 같다.

| split | samples | dialogues | label counts |
| --- | ---: | ---: | --- |
| train | 9989 | 1038 | neutral 4710, joy 1743, sadness 684, anger 1108, surprise 1205, fear 268, disgust 271 |
| val/dev | 1109 | 114 | neutral 470, joy 163, sadness 111, anger 153, surprise 150, fear 40, disgust 22 |
| test | 2610 | 280 | neutral 1256, joy 402, sadness 208, anger 345, surprise 281, fear 50, disgust 68 |

감정 클래스 인덱스 순서는 `Emotion` enum 순서다.

```text
0 neutral
1 joy
2 sadness
3 anger
4 surprise
5 fear
6 disgust
```

### RawSample 구성

`MeldDatasetSource`는 metadata row를 `RawSample`로 바꾼다.

- `uid`: `{split}:{dialogue_id}_{utterance_id}` 형식. 예: `train:0_0`
- `dialogue_id`: metadata의 `dialog`
- `utterance_id`: metadata의 `utterance`
- `text`: metadata의 `text`
- `speaker`: 빈 문자열 `""`
- `emotion`: metadata의 `y`
- `sentiment`: `None`
- `mask`: `ModalityMask.full()`
- `audio`: `AudioInput(sample_rate=16000)`, `source_path` 없음
- `video`: `VideoInput(fps=25.0)`, `source_path` 없음
- `metadata`: `source=meld_metadata`, `num_words`

중요한 제한: metadata 기반 로딩에서는 실제 화자명이 들어오지 않는다. 따라서 `dialogue_rnn`의 speaker embedding은 구조상 존재하지만, `ours_v1`에서는 모든 발화가 같은 빈 speaker로 취급된다.

### 특징 추출과 전처리

YAML의 extractor는 두 개다.

```yaml
extractors:
  - type: meld_precomputed
    path: MELD.Features.Models/features/text_glove_average_emotion.pkl
    modality: text
  - type: meld_precomputed
    path: MELD.Features.Models/features/audio_embeddings_feature_selection_emotion.pkl
    modality: audio
```

두 extractor 모두 `MeldPrecomputedFeatureExtractor`를 사용한다. pickle은 `(train, dev, test)` 세 split mapping을 담고 있어야 하며, 각 sample의 key는 우선 `{dialogue_id}_{utterance_id}`로 찾는다. 만약 dialogue 단위 matrix가 들어 있는 pickle이면 dialogue id로 matrix를 찾고 utterance id row를 선택하는 fallback도 있다.

확인된 feature 차원은 다음과 같다.

| feature file | modality | kind | split items | dim |
| --- | --- | --- | ---: | ---: |
| `text_glove_average_emotion.pkl` | text | embedding | train 9989, dev 1109, test 2610 | 300 |
| `audio_embeddings_feature_selection_emotion.pkl` | audio | embedding | train 9989, dev 1109, test 2610 | 1611 |

`FeaturePipeline`은 extractor별 `fit` 후 `transform`을 수행하고 `FeatureBundle`을 만든다. 이때 matrix는 sample 순서대로 `np.vstack`된다. `ours_v1`의 최종 early-fusion 입력 차원은 text 300 + audio 1611 = 1911이다.

raw MP4 lazy-load는 이 suite에서 사실상 발생하지 않는다. audio extractor가 있으므로 `FeaturePipeline`은 audio 필요 여부를 감지하지만, metadata 기반 `AudioInput`에는 `source_path`가 없고 precomputed extractor는 waveform을 요구하지 않는다. 따라서 PyAV/ffmpeg 기반 audio 로딩, frame 로딩, MFCC, 얼굴 landmark 같은 raw 전처리는 `ours_v1`에 포함되지 않는다.

캐시는 `cache.type: memory`라서 한 실행 안에서만 feature matrix를 재사용한다. 디스크 캐시 산출물은 없다.

### 모델 구성

`ours_v1` suite는 총 8개 실험을 비교한다.

| experiment | model type | base / 구조 |
| --- | --- | --- |
| `majority` | early fusion | 최빈 클래스 예측 |
| `random` | early fusion | seed 0 random probability |
| `early_centroid` | early fusion | nearest centroid, temperature 1.0 |
| `early_linear_regression` | early fusion | one-vs-rest ridge-style linear regression, alpha 0.001 |
| `early_logreg` | early fusion | StandardScaler + LogisticRegression, C 1.0, max_iter 1000 |
| `early_svm` | early fusion | StandardScaler + SVC, C 1.0, RBF kernel, probability=True |
| `late_centroid` | late fusion | modality별 centroid 학습 후 mean probability combiner |
| `dialogue_rnn` | dialogue-level classifier | PyTorch GRU + gated fusion + causal memory attention |

Early fusion은 `FeatureBundle.stack()`으로 모든 feature matrix를 concatenate한 뒤 하나의 estimator를 학습한다. `use_concepts: true`로 되어 있지만, `ours_v1` extractor는 모두 `embedding` kind라서 concept feature는 없다.

Late fusion은 모달리티별로 별도 estimator를 학습하고, 예측 시 각 모달리티 probability를 combiner가 합친다. `late_centroid`는 text centroid와 audio centroid를 따로 학습한 뒤 `MeanCombiner`를 사용한다.

### dialogue_rnn 구조

`dialogue_rnn`은 `TorchDialogueEmotionClassifier` adapter가 `FeatureBundle`을 dialogue batch tensor로 재구성한 뒤 `MultimodalEmotionModel`을 학습한다.

입력 재구성 방식:

- 발화를 `dialogue_id`별로 묶고 `utterance_id` 순서로 정렬한다.
- batch 단위는 sample이 아니라 dialogue다.
- 각 dialogue는 batch 내 max dialogue length까지 padding된다.
- `text_x`, `audio_x`, `video_x` shape는 `[B, N, 1, D]`다.
- `utterance_mask`는 padding 발화를 0으로 표시한다.
- `modality_mask`는 `[B, N, 3]`이며 text/audio/video 사용 가능 여부를 담는다.

`ours_v1`에는 video extractor가 없다. 그래서 video 입력 차원은 내부적으로 1로 보정되고 값은 0이며, `modality_mask[..., 2]`는 0이다. 모델 구조에는 video branch가 있지만 학습 신호는 text/audio에서만 온다.

모델 내부 모듈:

- `AttentiveRnnEncoder` 3개: text/audio/video 각각 `Linear(input_dim -> proj_dim)` + `LayerNorm` + dropout + GRU/LSTM + attention pooling
- `GatedMultimodalFusion`: text/audio/video embedding과 modality mask를 받아 gated sum을 만들고, interaction feature 사용 시 `text*audio`, `text*video`, `audio*video`도 concat
- `DialogueContextRnn`: fused utterance vector와 speaker embedding을 concat한 뒤 dialogue-level GRU/LSTM으로 문맥 인코딩
- `MemoryAttention`: 현재 발화가 자기 자신과 과거 발화에만 attend하는 causal attention. relative distance bias와 same-speaker bias 사용 가능
- `EmotionClassifierHead`: fused vector, context vector, memory vector를 concat해 7-class logits 출력

`ours_v1` YAML의 주요 hyperparameter:

| block | value |
| --- | --- |
| `rnn_type` | `gru` |
| modality encoder | `proj_dim=128`, `hidden_dim=128`, `dropout=0.2` |
| fusion | `fusion_dim=256`, `dropout=0.3`, gated fusion on, interaction features on |
| dialogue context | `speaker_emb_dim=32`, `hidden_dim=256`, `num_layers=1`, `dropout=0.3` |
| memory attention | enabled, `attn_dim=256`, RoPE off, relative distance bias on, same speaker bias on, max relative distance 32 |
| classifier head | `hidden_dim=256`, `dropout=0.3` |
| training | `lr=0.0002`, `weight_decay=0.01`, `gradient_clip_norm=1.0`, `batch_size=8`, `max_epochs=100`, `early_stopping_patience=10`, `validation_fraction=0.1`, `modality_dropout=0.1`, `seed=0`, `device=mps` |

학습 방식:

- optimizer: `torch.optim.AdamW`
- loss: `CrossEntropyLoss`
- class imbalance 대응: train label 빈도 기반 class weight 사용
- validation split: train dialogues를 seed 0으로 shuffle한 뒤 dialogue 단위 10%를 validation으로 사용
- early stopping 기준: validation weighted F1
- gradient clipping: norm 1.0
- training-time modality dropout: 각 발화/모달리티를 확률 0.1로 drop하되, 사용 가능한 모달리티가 모두 사라지면 첫 번째 available modality를 복구
- best validation state는 CPU state dict로 보관했다가 학습 종료 후 reload

현재 모델 checkpoint는 파일로 저장하지 않는다. 실행이 끝나면 JSON 결과만 남는다.

### 평가와 강건성 시나리오

평가 metric은 다음 네 개다.

```text
accuracy
macro_f1
weighted_f1
per_class_recall
```

confusion matrix도 저장한다. suite 비교표에는 `metrics: [accuracy, macro_f1, weighted_f1]`만 표시하고, robustness 비교 기준은 `weighted_f1`이다.

강건성 시나리오는 다음 세 개다.

```text
full
no_text
no_audio
```

`mask_bundle`은 제거된 모달리티의 feature matrix 값을 0으로 만들고 availability도 false로 바꾼다. 모델은 같은 학습된 checkpoint/estimator를 사용해 masked test bundle에서 다시 평가된다.

### 출력 구조와 결과물

주요 결과 파일은 다음이다.

```text
outputs/meld_raw_train_test_models.json
```

파일은 `ComparisonReport` 직렬화 결과다.

```text
{
  "name": "meld_raw_train_test_models",
  "outcomes": [
    {
      "name": "...",
      "result": {
        "name": "...",
        "evaluation": {
          "scenario": "full",
          "metrics": [...],
          "confusion": {"matrix": ..., "labels": [...]}
        },
        "robustness": {
          "reports": [
            {"scenario": "full", ...},
            {"scenario": "no_text", ...},
            {"scenario": "no_audio", ...}
          ]
        },
        "explanation": null,
        "metadata": {
          "classifier": "...",
          "n_train": "9989",
          "n_test": "2610",
          "train_split": "train",
          "eval_split": "test",
          "dropout": "none"
        }
      },
      "error": null
    }
  ]
}
```

suite는 일부 실험이 실패해도 전체 실행을 멈추지 않고 해당 outcome에 `error` 문자열을 기록한다. `ours_v1`의 현재 출력에서는 8개 실험이 모두 성공했다.

확인된 주요 test 성능은 다음과 같다.

| experiment | accuracy | macro_f1 | weighted_f1 |
| --- | ---: | ---: | ---: |
| majority | 0.4812 | 0.0928 | 0.3127 |
| random | 0.1414 | 0.1112 | 0.1697 |
| early_centroid | 0.2330 | 0.1762 | 0.2229 |
| early_linear_regression | 0.3747 | 0.2475 | 0.3817 |
| early_logreg | 0.2184 | 0.1863 | 0.2520 |
| early_svm | 0.5747 | 0.2839 | 0.5105 |
| late_centroid | 0.2785 | 0.2081 | 0.2907 |
| dialogue_rnn | 0.5211 | 0.3757 | 0.5403 |

강건성 weighted F1은 다음과 같다.

| experiment | full | no_text | no_audio |
| --- | ---: | ---: | ---: |
| majority | 0.3127 | 0.3127 | 0.3127 |
| random | 0.1697 | 0.1697 | 0.1697 |
| early_centroid | 0.2229 | 0.2677 | 0.0232 |
| early_linear_regression | 0.3817 | 0.3132 | 0.3657 |
| early_logreg | 0.2520 | 0.1936 | 0.3862 |
| early_svm | 0.5105 | 0.3323 | 0.0301 |
| late_centroid | 0.2907 | 0.1656 | 0.3396 |
| dialogue_rnn | 0.5403 | 0.0266 | 0.5289 |

해석 결과(`explainers`)는 YAML에서 비어 있으므로 생성되지 않는다. 즉 permutation importance, modality ablation explanation, counterfactual explanation은 `ours_v1` 결과 JSON에 포함되지 않는다. 강건성 평가는 있지만 explanation report는 없다.

### ours_v1의 명확한 한계

- raw MP4 기반 audio/video 전처리는 사용하지 않는다.
- video feature가 없다. `dialogue_rnn`의 video branch는 구조만 있고 실제 입력은 0/masked 상태다.
- metadata 로딩에서는 speaker가 빈 문자열이라 speaker-aware 모델의 장점이 제한된다.
- precomputed GloVe average와 audio embedding이 어떤 upstream 절차로 만들어졌는지는 이 프로젝트 내부에서 재현하지 않는다.
- 모델 checkpoint, per-sample prediction, attention/gate 값은 파일로 저장하지 않는다.
- suite-level `dropout`은 설정하지 않았다. `dialogue_rnn` 내부 training modality dropout만 0.1로 사용한다.
- explanation pipeline은 구현되어 있지만 `ours_v1` suite에서는 켜지 않았다.

### ours_v2 개발 참고

`ours_v2`에서 가장 자연스럽게 이어갈 수 있는 축은 다음이다.

1. video modality 추가: MELD precomputed video feature를 붙이거나 raw MP4에서 얼굴/프레임 feature를 구현해 `extractors`에 `video`를 추가한다.
2. speaker 정보 복원: metadata 대신 CSV를 기준으로 speaker를 살리되 precomputed feature와 split/key alignment를 유지하는 dataset source 또는 metadata 확장을 만든다.
3. 결과 저장 확장: model checkpoint, per-sample prediction, modality gate, memory attention, confusion matrix 요약을 별도 artifact로 저장한다.
4. explanation 활성화: `PermutationConfig`, `ModalityAblationConfig`, `CounterfactualConfig`를 suite에 추가해 해석 결과를 같이 남긴다.
5. dialogue model 튜닝: validation split을 dev split으로 분리하거나, modality dropout, class weight, fusion interaction, memory attention 옵션을 ablation한다.
6. raw feature 재현성 강화: placeholder 상태인 TF-IDF, sentence embedding, MFCC, visual cue를 실제 구현으로 대체하면 precomputed 의존도를 줄일 수 있다.
