"""EG-CB-MER · 멀티모달 감정 분석 데모.

영상(또는 텍스트/오디오) 하나를 입력하면 감정을 예측하고,
어떤 단서가 예측에 기여했는지 설명합니다.

실행: python app.py   →   http://localhost:7860
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import textwrap
import warnings

import cv2
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from scipy.io import wavfile

from meld_emotion.core.data import AudioInput, ModalityMask, RawSample, VideoInput
from meld_emotion.core.features import FeatureBundle, FeatureKind
from meld_emotion.core.types import EMOTION_ORDER, Emotion, Modality, Split
from meld_emotion.data.labels import EmotionLabelEncoder
from meld_emotion.data.synthetic import SyntheticDatasetSource
from meld_emotion.features.audio import AudioConceptExtractor
from meld_emotion.features.text import BowTextExtractor, TextConceptExtractor
from meld_emotion.features.video import VideoConceptExtractor
from meld_emotion.fusion.early import EarlyFusionClassifier
from meld_emotion.fusion.masking import ModalityScenario, mask_bundle
from meld_emotion.models.sklearn_estimators import LogisticRegressionEstimator
from meld_emotion.pipeline.cache import NullFeatureCache
from meld_emotion.pipeline.feature_pipeline import FeaturePipeline

# ── 상수 ────────────────────────────────────────────────────────────────────────

EMOTION_EMOJI: dict[str, str] = {
    "neutral": "😐", "joy": "😄", "sadness": "😢",
    "anger": "😠", "surprise": "😲", "fear": "😨", "disgust": "🤢",
}
EMOTION_COLOR: dict[str, str] = {
    "neutral": "#7F8C8D", "joy": "#F39C12", "sadness": "#2980B9",
    "anger": "#E74C3C", "surprise": "#8E44AD", "fear": "#16A085", "disgust": "#27AE60",
}
MODALITY_COLOR = {Modality.TEXT: "#4C72B0", Modality.AUDIO: "#DD8452", Modality.VIDEO: "#55A868"}

_N_FRAMES = 8
_FRAME_H, _FRAME_W = 64, 64

# ── 모델 초기화 (앱 시작 시 1회) ────────────────────────────────────────────────

_pipeline: FeaturePipeline
_classifier: EarlyFusionClassifier
_concept_means: dict[str, np.ndarray]   # source → (n_features,) 학습 평균
_encoder = EmotionLabelEncoder()


def _init_model() -> None:
    global _pipeline, _classifier, _concept_means

    print("모델 초기화 중...", end=" ", flush=True)
    source = SyntheticDatasetSource(n_train=700, n_dev=0, n_test=0, seed=42)
    train_samples = list(source.load(Split.TRAIN))

    _pipeline = FeaturePipeline(
        [
            TextConceptExtractor(),
            BowTextExtractor(n_features=256),
            AudioConceptExtractor(),
            VideoConceptExtractor(),
        ],
        NullFeatureCache(),
    )

    train_bundle = _pipeline.fit_transform(train_samples, Split.TRAIN)
    y_train = _encoder.encode([s.emotion for s in train_samples])

    _classifier = EarlyFusionClassifier(
        lambda: LogisticRegressionEstimator(C=1.0, max_iter=1000),
        _encoder.classes,
        use_concepts=True,
    )
    _classifier.fit(train_bundle, y_train)

    # 반사실 설명용: 개념 특징의 학습 평균 저장
    _concept_means = {}
    for mat in train_bundle.matrices:
        if mat.kind == FeatureKind.CONCEPT:
            _concept_means[mat.source] = mat.values.mean(axis=0)

    print("완료 ✓")


_init_model()


# ── 미디어 로더 ─────────────────────────────────────────────────────────────────

def _load_audio(path: str) -> AudioInput | None:
    """wav 파일을 AudioInput 으로 변환."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sr, data = wavfile.read(path)
        data = data.astype(np.float64)
        if data.ndim > 1:
            data = data.mean(axis=1)
        max_val = np.max(np.abs(data))
        if max_val > 1.0:
            data /= max_val if max_val > 0 else 1.0
        return AudioInput(sample_rate=int(sr), waveform=data)
    except Exception:
        return None


def _load_video_frames(path: str) -> VideoInput | None:
    """mp4 등 영상에서 프레임을 균등 샘플링해 VideoInput 으로 변환."""
    try:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if total <= 0:
            cap.release()
            return None
        indices = np.linspace(0, max(total - 1, 0), _N_FRAMES, dtype=int)
        frames: list[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.resize(frame, (_FRAME_W, _FRAME_H))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.astype(np.float64) / 255.0)
        cap.release()
        if not frames:
            return None
        return VideoInput(fps=fps, frames=np.stack(frames, axis=0))
    except Exception:
        return None


def _extract_audio_from_video(path: str) -> AudioInput | None:
    """영상에서 오디오 트랙을 추출 (ffmpeg 필요, 없으면 None)."""
    try:
        import subprocess, tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", "16000",
             "-vn", tmp_path],
            capture_output=True, timeout=30,
        )
        if result.returncode == 0:
            audio = _load_audio(tmp_path)
            os.unlink(tmp_path)
            return audio
        os.unlink(tmp_path)
    except Exception:
        pass
    return None


# ── 분석 코어 ────────────────────────────────────────────────────────────────────

def _build_sample(text: str, audio: AudioInput | None, video: VideoInput | None) -> RawSample:
    available: set[Modality] = {Modality.TEXT}
    if audio is not None:
        available.add(Modality.AUDIO)
    if video is not None:
        available.add(Modality.VIDEO)
    return RawSample(
        uid="user_input",
        dialogue_id=0,
        utterance_id=0,
        text=text,
        speaker="User",
        split=Split.TEST,
        mask=ModalityMask(available=frozenset(available)),
        audio=audio,
        video=video,
    )


def _modality_contributions(
    bundle: FeatureBundle,
    pred_cls: int,
) -> dict[Modality, float]:
    """단일 샘플 모달리티 기여도 (예측 클래스 확률 하락량)."""
    base_prob = float(_classifier.predict_proba(bundle)[0, pred_cls])

    drops: dict[Modality, float] = {}
    all_mod = frozenset(Modality)
    for mod in Modality:
        ablated = mask_bundle(
            bundle,
            ModalityScenario(f"no_{mod.value}", all_mod - {mod}),
        )
        ablated_prob = float(_classifier.predict_proba(ablated)[0, pred_cls])
        drops[mod] = max(base_prob - ablated_prob, 0.0)

    total = sum(drops.values()) or 1.0
    return {m: v / total for m, v in drops.items()}


def _counterfactual(
    bundle: FeatureBundle,
    pred_cls: int,
    top_k: int = 3,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """개념 특징 top-k 를 학습 평균으로 대체했을 때 확률 변화."""
    orig_proba = _classifier.predict_proba(bundle)[0]

    concept_indices = [
        (i, mat)
        for i, mat in enumerate(bundle.matrices)
        if mat.kind == FeatureKind.CONCEPT
    ]
    if not concept_indices:
        return orig_proba, orig_proba.copy(), []

    # 이탈도(deviation) 계산
    scored: list[tuple[float, int, int, str]] = []
    for mat_idx, mat in concept_indices:
        mean = _concept_means.get(mat.source, np.zeros(mat.n_features))
        for col in range(mat.n_features):
            dev = abs(float(mat.values[0, col]) - float(mean[col]))
            scored.append((dev, mat_idx, col, mat.names[col]))
    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[:top_k]

    # 복사본에 평균값 적용
    matrices = [m.values.copy() for m in bundle.matrices]
    removed_names: list[str] = []
    for _, mat_idx, col, name in top:
        mat = bundle.matrices[mat_idx]
        mean = _concept_means.get(mat.source, np.zeros(mat.n_features))
        matrices[mat_idx][0, col] = mean[col]
        removed_names.append(name)

    from meld_emotion.core.features import FeatureMatrix
    rebuilt_matrices = tuple(
        FeatureMatrix(
            values=matrices[i],
            names=m.names,
            modality=m.modality,
            kind=m.kind,
            source=m.source,
        )
        for i, m in enumerate(bundle.matrices)
    )
    mod_bundle = FeatureBundle(
        uids=bundle.uids,
        matrices=rebuilt_matrices,
        availability=bundle.availability,
    )
    mod_proba = _classifier.predict_proba(mod_bundle)[0]
    return orig_proba, mod_proba, removed_names


# ── 시각화 ───────────────────────────────────────────────────────────────────────

def _fig_proba(proba: np.ndarray) -> plt.Figure:
    """7개 감정 확률 수평 막대 그래프."""
    emotions = [e.value for e in EMOTION_ORDER]
    colors = [EMOTION_COLOR[e] for e in emotions]
    order = np.argsort(proba)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(
        [emotions[i] for i in order],
        [proba[i] for i in order],
        color=[colors[i] for i in order],
        edgecolor="white",
        height=0.6,
    )
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Probability", fontsize=11)
    ax.set_title("Emotion Probability Distribution", fontsize=12, fontweight="bold")
    ax.axvline(1 / 7, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="random baseline")
    ax.legend(fontsize=8)
    for bar, idx in zip(bars, order):
        ax.text(proba[idx] + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{proba[idx]:.3f}", va="center", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def _fig_modality(contrib: dict[Modality, float], available: set[Modality]) -> plt.Figure:
    """모달리티 기여도 파이 + 막대."""
    labels = [m.value.capitalize() for m in Modality]
    values = [contrib.get(m, 0.0) for m in Modality]
    colors = [MODALITY_COLOR[m] for m in Modality]
    # 사용 불가 모달리티는 빗금 표시
    hatches = ["" if m in available else "///" for m in Modality]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    # 막대
    x = np.arange(len(labels))
    for xi, (v, c, h, lbl) in enumerate(zip(values, colors, hatches, labels)):
        ax1.bar(xi, v, color=c, hatch=h, edgecolor="white", width=0.5, alpha=0.9 if h == "" else 0.4)
        ax1.text(xi, v + 0.005, f"{v * 100:.1f}%", ha="center", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylim(0, max(values) * 1.3 + 0.05)
    ax1.set_ylabel("Contribution (normalized)", fontsize=10)
    ax1.set_title("Modality Contribution\n(ablation-based)", fontsize=11, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # 파이
    pie_vals = [max(v, 0.001) for v in values]
    wedges, texts, autotexts = ax2.pie(
        pie_vals, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 10},
    )
    for wdg, h in zip(wedges, hatches):
        if h:
            wdg.set_hatch(h)
            wdg.set_alpha(0.4)
    ax2.set_title("Relative Share", fontsize=11, fontweight="bold")

    # 범례: 빗금 = 모달리티 없음
    if any(h for h in hatches):
        from matplotlib.patches import Patch
        ax2.legend(
            handles=[Patch(facecolor="gray", hatch="///", alpha=0.4, label="Not provided")],
            fontsize=8, loc="lower right",
        )

    plt.tight_layout()
    return fig


def _fig_counterfactual(orig: np.ndarray, mod: np.ndarray) -> plt.Figure:
    """반사실 전후 확률 비교 막대 그래프."""
    emotions = [e.value for e in EMOTION_ORDER]
    x = np.arange(len(emotions))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, orig, w, label="Original", color="#4C72B0", edgecolor="white", alpha=0.9)
    ax.bar(x + w / 2, mod, w, label="After removal", color="#E74C3C", edgecolor="white", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(emotions, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Probability", fontsize=11)
    ax.set_title("Counterfactual — What if we remove top evidence?", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def _concept_feature_fig(sample: RawSample, bundle: FeatureBundle) -> plt.Figure:
    """텍스트 + 오디오 + 비디오 개념 특징을 한 화면에 표시."""
    concept_mats = [m for m in bundle.matrices if m.kind == FeatureKind.CONCEPT]
    if not concept_mats:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "개념 특징 없음", ha="center", va="center")
        return fig

    n_mats = len(concept_mats)
    fig, axes = plt.subplots(1, n_mats, figsize=(4.5 * n_mats, 4))
    if n_mats == 1:
        axes = [axes]

    for ax, mat in zip(axes, concept_mats):
        values = mat.values[0]
        names = list(mat.names)
        color = MODALITY_COLOR.get(mat.modality, "#888")

        bars = ax.barh(names, values, color=color, edgecolor="white", height=0.6, alpha=0.85)
        ax.set_title(f"{mat.modality.value.capitalize()} Concepts", fontsize=11, fontweight="bold")
        ax.set_xlabel("Value", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, val in zip(bars, values):
            ax.text(max(val, 0) + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)

        if mat.modality not in sample.mask.available:
            ax.set_facecolor("#f8f0f0")
            ax.text(0.5, 0.5, "모달리티 없음\n(0 벡터)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="gray",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    plt.tight_layout()
    return fig


def _frame_grid(video: VideoInput | None) -> plt.Figure | None:
    """비디오 프레임 그리드 (최대 8개)."""
    if video is None or video.frames is None:
        return None
    frames = video.frames  # (T, H, W, C), 0~1
    T = min(frames.shape[0], 8)
    cols = 4
    rows = (T + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes_flat = np.array(axes).flatten()
    for i in range(cols * rows):
        ax = axes_flat[i]
        if i < T:
            ax.imshow(np.clip(frames[i], 0, 1))
            ax.set_title(f"f{i}", fontsize=7)
        ax.axis("off")
    fig.suptitle("추출된 비디오 프레임", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


# ── 메인 분석 함수 ───────────────────────────────────────────────────────────────

def analyze(
    text: str,
    audio_path: str | None,
    video_path: str | None,
) -> tuple:
    """사용자 입력 → 분석 결과 반환."""

    text = (text or "").strip()
    if not text:
        err = "⚠️ 텍스트를 입력해 주세요."
        return err, None, None, None, None, None, None

    # 미디어 로드
    audio: AudioInput | None = None
    video: VideoInput | None = None

    if video_path:
        video = _load_video_frames(video_path)
        if audio is None:          # 영상에서 오디오 추출 시도
            audio = _extract_audio_from_video(video_path)

    if audio_path:                 # 별도 오디오 파일이 있으면 우선 사용
        loaded = _load_audio(audio_path)
        if loaded is not None:
            audio = loaded

    # RawSample 구성
    sample = _build_sample(text, audio, video)
    available = sample.mask.available

    # 특징 추출
    bundle = _pipeline.transform([sample], Split.TEST)

    # 예측
    proba = _classifier.predict_proba(bundle)[0]
    pred_cls = int(np.argmax(proba))
    pred_emotion = EMOTION_ORDER[pred_cls]
    confidence = float(proba[pred_cls])

    # 모달리티 기여도
    contrib = _modality_contributions(bundle, pred_cls)

    # 반사실
    orig_proba, mod_proba, removed = _counterfactual(bundle, pred_cls, top_k=3)
    cf_delta = float(proba[pred_cls]) - float(mod_proba[pred_cls])

    # ── 결과 조립 ──────────────────────────────────────────────────────────────

    # (1) 감정 예측 마크다운 배지
    emoji = EMOTION_EMOJI[pred_emotion.value]
    color = EMOTION_COLOR[pred_emotion.value]
    mod_labels = " · ".join(m.value.upper() for m in sorted(available, key=lambda x: x.value))
    badge_md = textwrap.dedent(f"""
    <div style="
        background:{color}22;
        border-left:6px solid {color};
        border-radius:8px;
        padding:18px 22px;
        margin-bottom:8px;
    ">
      <div style="font-size:2.4rem;line-height:1">{emoji}</div>
      <div style="font-size:1.8rem;font-weight:700;color:{color}">{pred_emotion.value.upper()}</div>
      <div style="font-size:1.1rem;margin-top:4px">신뢰도 <b>{confidence:.1%}</b></div>
      <div style="font-size:0.9rem;color:#666;margin-top:6px">사용된 모달리티: {mod_labels}</div>
    </div>
    """).strip()

    # (2) 반사실 요약 마크다운
    cf_md = textwrap.dedent(f"""
    ### 반사실 설명 (Counterfactual)

    모델이 가장 중요하게 본 개념 단서 **{len(removed)}개**를 학습 평균값으로 대체했습니다.

    | | 예측 감정 | {pred_emotion.value} 확률 |
    |---|---|---|
    | **원래** | **{pred_emotion.value}** ({confidence:.3f}) | {proba[pred_cls]:.3f} |
    | **단서 제거 후** | **{EMOTION_ORDER[int(np.argmax(mod_proba))].value}** ({np.max(mod_proba):.3f}) | {mod_proba[pred_cls]:.3f} |

    **확률 하락: {cf_delta:+.3f}**
    {'→ 이 단서들이 예측에 실질적으로 기여했습니다.' if cf_delta > 0.02 else '→ 예측이 비교적 다른 단서들에도 분산되어 있습니다.'}

    **제거된 증거 (Top-{len(removed)})**: {', '.join(f'`{r}`' for r in removed) or '없음'}
    """).strip()

    # 시각화
    fig_proba = _fig_proba(proba)
    fig_mod = _fig_modality(contrib, available)
    fig_concept = _concept_feature_fig(sample, bundle)
    fig_cf = _fig_counterfactual(orig_proba, mod_proba)
    fig_frames = _frame_grid(video)

    return badge_md, fig_proba, fig_mod, fig_concept, fig_cf, cf_md, fig_frames


# ── Gradio UI ───────────────────────────────────────────────────────────────────

EXAMPLE_TEXTS = [
    "I can't believe you actually did that! That's absolutely unacceptable!",
    "I miss you so much. Nothing feels the same without you.",
    "Oh my god, are you serious?! That's incredible news!",
    "Sure, I'll take care of it. No problem at all.",
    "This is disgusting. How could anyone think this is okay?",
    "I'm so scared. Please, just let me go.",
    "We did it! I'm so happy right now, this is amazing!",
]

with gr.Blocks(title="EG-CB-MER · 멀티모달 감정 분석") as demo:
    gr.HTML("""
    <div style="text-align:center;padding:20px 0 8px">
      <h1 style="font-size:1.8rem;margin:0">🎭 멀티모달 감정 분석 데모</h1>
      <p style="color:#666;margin:6px 0 0">
        텍스트 · 오디오 · 비디오를 입력하면 감정을 예측하고<br>
        <b>어떤 단서</b>가 예측에 기여했는지 설명합니다.
      </p>
    </div>
    """)

    with gr.Row(equal_height=False):
        # ── 입력 패널 ─────────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("## 📥 입력")

            text_input = gr.Textbox(
                label="발화 텍스트 (필수)",
                placeholder="분석할 문장을 입력하세요...\n예: I can't believe you did that!",
                lines=4,
                max_lines=8,
            )

            with gr.Row():
                audio_input = gr.Audio(
                    label="오디오 파일 (선택, .wav)",
                    type="filepath",
                    sources=["upload"],
                )

            video_input = gr.Video(
                label="영상 파일 (선택, .mp4 / .avi / .mov)",
                sources=["upload"],
            )

            analyze_btn = gr.Button("분석하기 ▶", variant="primary", size="lg")

            gr.Markdown("### 예시 문장")
            gr.Examples(
                examples=[[t] for t in EXAMPLE_TEXTS],
                inputs=[text_input],
                label="",
            )

        # ── 결과 패널 ─────────────────────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("## 📊 분석 결과")

            emotion_badge = gr.HTML(
                value="<div style='color:#aaa;padding:20px;text-align:center'>"
                      "텍스트를 입력하고 <b>분석하기</b>를 클릭하세요.</div>"
            )

            with gr.Tabs():
                with gr.Tab("감정 확률"):
                    gr.Markdown("7개 감정 클래스의 예측 확률. 수평선은 무작위 기준선(1/7≈14%).")
                    plot_proba = gr.Plot(label="")

                with gr.Tab("모달리티 기여도"):
                    gr.Markdown(
                        "각 모달리티를 제거했을 때 **예측 확률 하락폭**으로 기여도를 측정합니다.  \n"
                        "빗금(//) 모달리티는 이번 입력에서 제공되지 않은 것입니다."
                    )
                    plot_modality = gr.Plot(label="")

                with gr.Tab("개념 특징 분석"):
                    gr.Markdown(
                        "텍스트·오디오·비디오 각 모달리티에서 추출한 **해석 가능한 개념 특징** 값.  \n"
                        "높은 값 = 해당 단서가 강하게 존재함."
                    )
                    plot_concept = gr.Plot(label="")

                with gr.Tab("반사실 설명"):
                    cf_text = gr.Markdown()
                    plot_cf = gr.Plot(label="")

                with gr.Tab("비디오 프레임"):
                    gr.Markdown("업로드된 영상에서 균등하게 샘플링한 프레임.")
                    plot_frames = gr.Plot(label="")

    # ── 이벤트 ───────────────────────────────────────────────────────────────────
    analyze_btn.click(
        fn=analyze,
        inputs=[text_input, audio_input, video_input],
        outputs=[
            emotion_badge,
            plot_proba,
            plot_modality,
            plot_concept,
            plot_cf,
            cf_text,
            plot_frames,
        ],
    )

    # 텍스트 입력 후 Enter 없이도 즉시 분석 (선택)
    text_input.submit(
        fn=analyze,
        inputs=[text_input, audio_input, video_input],
        outputs=[
            emotion_badge,
            plot_proba,
            plot_modality,
            plot_concept,
            plot_cf,
            cf_text,
            plot_frames,
        ],
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Gradio 공개 링크 생성 (72h)")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
    )
