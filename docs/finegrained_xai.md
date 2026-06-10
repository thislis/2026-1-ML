# Fine-Grained Dialogue XAI

`dialogue_finegrained_xai` 는 `dialogue_rnn` 예측 하나를 target 으로 잡고, 단어/token,
오디오 시간 구간, 비디오 프레임, source utterance, modality, classifier block 중요도를 한 번에
저장한다.

## 데이터 흐름

기존 pooled extractor 는 발화 하나를 `(D,)` 벡터로 요약한다. fine-grained XAI 는 발화 내부 위치를
보존해야 하므로 새 sequence extractor 를 쓴다.

```text
RawSample
  -> text_token_embeddings              [n, L_T, D_T]
  -> audio_wav2vec2_xlsr_sequence       [n, L_A, D_A]
  -> video_frame_embeddings             [n, L_V, D_V]
  -> FeatureBundle.sequence_matrices
  -> TorchDialogueEmotionClassifier     [B, N, L, D]
  -> dialogue_finegrained_xai
```

`FeatureUnit` 은 각 sequence 위치의 사람이 읽을 수 있는 metadata 를 보존한다.

- text: token label, character span
- audio: hidden step 이 대응하는 start/end seconds
- video: sampled frame index, timestamp

## Attribution 집계

Captum Integrated Gradients 는 target logit `logits[b,t,c]` 에 대해 다음 attribution 을 만든다.

```text
attr_text:  [B, N, L_T, D_T]
attr_audio: [B, N, L_A, D_A]
attr_video: [B, N, L_V, D_V]
```

같은 attribution 을 여러 해상도로 합산한다.

- token/span/frame 중요도: `sum_abs(attr[b,j,l,:])`
- source utterance 중요도: `sum_abs(attr_text/audio/video[b,j,:,:])`
- modality share: 각 modality attribution 총합을 정규화
- embedding dimension attribution: `sum_abs(attr[..., d])`

모델 내부 관찰값도 함께 저장한다.

- `modality_gate`
- `text/audio/video_attention`
- `memory_attention`
- `u_text/u_audio/u_video`
- `fused/context_h/memory`

추가로 classifier head 입력 block ablation 을 계산한다.

```text
fused_removed   = score_orig - score_with_classifier_fused_zeroed
context_removed = score_orig - score_with_classifier_context_zeroed
memory_removed  = score_orig - score_with_classifier_memory_zeroed
```

## 실행 예시

```bash
uv sync --extra text --extra audio --extra video --extra deep --extra xai
uv run meld-emotion run --config configs/example_finegrained_xai.yaml
```

출력은 기본적으로 다음 파일에 저장된다.

- `outputs/finegrained_xai.json`: 전체 `ExperimentResult`
- `outputs/finegrained_xai_dashboard.json`: dashboard frontend 용 payload

단일 MP4+텍스트 inference 에서도 XAI 를 함께 볼 수 있다.

```bash
uv run meld-emotion infer \
  --mp4 sample.mp4 \
  --text "I am so happy!" \
  --checkpoint outputs/best_model.pt \
  --xai \
  --json \
  --xai-dashboard outputs/infer_xai_dashboard.json
```

`--xai` inference 는 기본 pooled inference extractor 대신 sequence extractor 를 사용한다. 따라서
checkpoint 의 text/audio/video input dim 이 기본 sequence extractor 출력 차원(768/1024/768)과
맞아야 한다. `--json` 출력의 `xai` 필드에는 전체 fine-grained result 가 들어가고,
`--xai-dashboard` 는 dashboard frontend 용 단일 inference payload 를 별도로 저장한다.

## Dashboard Payload

Dashboard exporter 는 실제 HTML 렌더링이 아니라 다음 패널을 만들 수 있는 JSON data contract 를
저장한다.

- target utterance: `uid`, `speaker`, `pred_class`, `pred_proba`
- modality panel: gate, attribution share, logit ablation delta
- dialogue panel: source utterance attribution share, memory attention
- block panel: `fused`, `context`, `memory`
- text/audio/video panel: top token, audio span, video frame
- dimension panel: modality별 top embedding dimensions

## 해석 주의사항

Attention 값은 모델 내부 관찰값이며, 예측 기여도 자체와 항상 같지는 않다. 보고서에서는
Integrated Gradients, modality/block ablation 과 함께 비교해서 해석해야 한다.

v1 비디오는 프레임 단위 중요도를 제공한다. 화면 영역, 얼굴 부위, patch heatmap 은 아직 구현하지
않았다. 이 후속 경로는 `version_history.md` 의 `ours_v2.5` 변천사와 후보 항목에 남겨 두었다.
