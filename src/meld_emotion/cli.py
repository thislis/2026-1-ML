"""명령줄 인터페이스.

- ``meld-emotion run --config <path>`` : YAML 설정으로 실험을 실행한다.
- ``meld-emotion infer --mp4 <path> --text <text>`` : 저장된 checkpoint 로 단일 입력을 추론한다.
- ``meld-emotion infer-batch --csv <path> --mp4-dir <dir>`` : MELD split 전체를 추론하고
  fine-grained XAI 분석 리포트를 만든다.
- ``meld-emotion status`` : 모든 컴포넌트의 구현 상태(REAL/PLACEHOLDER/UNIMPLEMENTED)를
  코드에서 직접 읽어 출력한다(할 일 목록의 단일 진실 공급원).
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

# builder 를 import 하면 모든 구체 컴포넌트가 로드되어 상태 레지스트리가 채워진다.
from meld_emotion.config.loader import load_config, load_suite
from meld_emotion.core.status import ComponentStatus, iter_status
from meld_emotion.logging_config import configure_logging
from meld_emotion.pipeline import builder
from meld_emotion.pipeline.suite import SuiteRunner
from meld_emotion.reporting.report import ComparisonReporter

logger = logging.getLogger(__name__)


def _force_utf8() -> None:
    """Windows 콘솔(cp949)에서도 한글/기호가 깨지지 않도록 UTF-8 로 강제한다."""

    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")


def _cmd_run(config_path: str, log_level: str, log_file: str | None) -> int:
    configure_logging(log_level, log_file)
    logger.info("단일 실험 실행 준비: config=%s", config_path)
    config = load_config(config_path)
    runner = builder.build_experiment(config)
    runner.run()
    logger.info("단일 실험 실행 완료: %s", config.name)
    return 0


def _cmd_compare(config_path: str, log_level: str, log_file: str | None) -> int:
    configure_logging(log_level, log_file)
    logger.info("비교 suite 실행 준비: config=%s", config_path)
    suite = load_suite(config_path)
    report = SuiteRunner(suite.name, suite.experiments).run()
    ComparisonReporter(
        metrics=suite.metrics,
        robustness_metric=suite.robustness_metric,
        path=suite.output_path,
    ).save(report)
    logger.info("비교 suite 실행 완료: %s", suite.name)
    return 0


def _cmd_status() -> int:
    records = list(iter_status())
    by_status: dict[ComponentStatus, list[str]] = {s: [] for s in ComponentStatus}
    for record in records:
        label = record.qualname.removeprefix("meld_emotion.")
        suffix = f"  — {record.reason}" if record.reason else ""
        by_status[record.status].append(f"  {label}{suffix}")

    order = [
        ComponentStatus.REAL,
        ComponentStatus.PLACEHOLDER,
        ComponentStatus.UNIMPLEMENTED,
    ]
    for status in order:
        items = by_status[status]
        print(f"[{status.value.upper()}] ({len(items)})")
        for line in sorted(items):
            print(line)
        print()
    total = len(records)
    done = len(by_status[ComponentStatus.REAL])
    print(
        f"요약: 전체 {total}개 중 완전구현 {done}, "
        f"임시 {len(by_status[ComponentStatus.PLACEHOLDER])}, "
        f"미구현 {len(by_status[ComponentStatus.UNIMPLEMENTED])}"
    )
    return 0


def _cmd_infer(
    mp4_path: str,
    text: str,
    checkpoint: str,
    device: str,
    top_k: int,
    json_output: bool,
    include_xai: bool,
    xai_steps: int,
    xai_top_k: int,
    xai_audio_window_seconds: float,
    xai_video_window_seconds: float,
    xai_max_units_per_modality: int,
    xai_dashboard: str | None,
    markdown_output: str | None,
    log_level: str,
    log_file: str | None,
) -> int:
    configure_logging(log_level, log_file)
    from meld_emotion.inference import (
        dashboard_to_json,
        format_inference_result,
        result_to_json,
        result_to_markdown,
        run_inference,
    )

    logger.info("단일 입력 추론 준비: mp4=%s checkpoint=%s", mp4_path, checkpoint)
    result = run_inference(
        mp4_path=mp4_path,
        text=text,
        checkpoint_path=checkpoint,
        device=device,
        top_k=top_k,
        include_xai=include_xai,
        xai_n_steps=xai_steps,
        xai_top_k=xai_top_k,
        xai_audio_window_seconds=xai_audio_window_seconds,
        xai_video_window_seconds=xai_video_window_seconds,
        xai_max_units_per_modality=xai_max_units_per_modality,
    )
    if xai_dashboard is not None:
        dashboard_path = Path(xai_dashboard)
        dashboard_path.parent.mkdir(parents=True, exist_ok=True)
        dashboard_path.write_text(dashboard_to_json(result), encoding="utf-8")
        logger.info("inference XAI dashboard 저장 완료: path=%s", dashboard_path)
    if markdown_output is not None:
        markdown_path = Path(markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(result_to_markdown(result), encoding="utf-8")
        logger.info("inference Markdown 설명 저장 완료: path=%s", markdown_path)
    print(result_to_json(result) if json_output else format_inference_result(result))
    logger.info(
        "단일 입력 추론 완료: label=%s probability=%.6f", result.label.value, result.probability
    )
    return 0


def _cmd_infer_batch(
    csv_path: str,
    mp4_dir: str,
    checkpoint: str,
    device: str,
    predictions_path: str,
    summary_path: str,
    report_path: str,
    manifest_path: str | None,
    suite_path: str,
    xai_steps: int,
    xai_top_k: int,
    resume: bool,
    limit: int | None,
    duplicate_uid_policy: str,
    log_level: str,
    log_file: str | None,
) -> int:
    configure_logging(log_level, log_file)
    from meld_emotion.inference_batch import run_batch_inference

    logger.info(
        "batch inference 준비: csv=%s mp4_dir=%s checkpoint=%s",
        csv_path,
        mp4_dir,
        checkpoint,
    )
    result = run_batch_inference(
        csv_path=csv_path,
        mp4_dir=mp4_dir,
        checkpoint_path=checkpoint,
        device=device,
        predictions_path=predictions_path,
        summary_path=summary_path,
        report_path=report_path,
        manifest_path=manifest_path,
        suite_path=suite_path,
        xai_steps=xai_steps,
        xai_top_k=xai_top_k,
        resume=resume,
        limit=limit,
        duplicate_uid_policy=duplicate_uid_policy,
    )
    print(f"predictions: {result.paths.predictions}")
    print(f"summary: {result.paths.summary}")
    print(f"report: {result.paths.report}")
    print(f"manifest: {result.paths.manifest}")
    logger.info("batch inference 완료: records=%s", result.summary.get("n_records"))
    return 0


def _cmd_infer_svm_batch(
    csv_path: str,
    mp4_dir: str,
    checkpoint: str,
    device: str,
    predictions_path: str,
    summary_path: str,
    top_k: int,
    xai_top_k: int,
    xai_audio_window_seconds: float,
    xai_video_window_seconds: float,
    xai_max_units_per_modality: int,
    resume: bool,
    limit: int | None,
    log_level: str,
    log_file: str | None,
) -> int:
    configure_logging(log_level, log_file)
    from meld_emotion.inference_svm_batch import run_svm_batch_inference

    logger.info(
        "SVM batch inference 준비: csv=%s mp4_dir=%s checkpoint=%s",
        csv_path,
        mp4_dir,
        checkpoint,
    )
    result = run_svm_batch_inference(
        csv_path=csv_path,
        mp4_dir=mp4_dir,
        checkpoint_path=checkpoint,
        device=device,
        predictions_path=predictions_path,
        summary_path=summary_path,
        top_k=top_k,
        xai_top_k=xai_top_k,
        xai_audio_window_seconds=xai_audio_window_seconds,
        xai_video_window_seconds=xai_video_window_seconds,
        xai_max_units_per_modality=xai_max_units_per_modality,
        resume=resume,
        limit=limit,
    )
    print(f"predictions: {result.paths.predictions}")
    print(f"summary: {result.paths.summary}")
    print(f"records: {result.summary.get('n_records')}")
    logger.info("SVM batch inference 완료: records=%s", result.summary.get("n_records"))
    return 0


def _cmd_fine_tune_embeddinggemma(
    csv_path: str,
    model_name: str,
    output_dir: str,
    epochs: float,
    batch_size: int,
    learning_rate: float,
    warmup_ratio: float,
    eval_fraction: float,
    seed: int,
    device: str,
    fp16: bool,
    bf16: bool,
    max_steps: int | None,
    save_total_limit: int,
    eval_steps: int,
    early_stopping_metric: str,
    early_stopping_patience: int,
    early_stopping_threshold: float,
    log_level: str,
    log_file: str | None,
) -> int:
    configure_logging(log_level, log_file)
    from meld_emotion.fine_tunning.embeddinggemma import (
        EmbeddingGemmaFineTuneConfig,
        run_embeddinggemma_fine_tuning,
    )

    logger.info("EmbeddingGemma fine-tuning 준비: csv=%s model=%s", csv_path, model_name)
    summary = run_embeddinggemma_fine_tuning(
        EmbeddingGemmaFineTuneConfig(
            csv_path=Path(csv_path),
            model_name=model_name,
            output_dir=Path(output_dir),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            eval_fraction=eval_fraction,
            seed=seed,
            device=device,
            fp16=fp16,
            bf16=bf16,
            max_steps=max_steps,
            save_total_limit=save_total_limit,
            eval_steps=eval_steps,
            early_stopping_metric=early_stopping_metric,
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
        )
    )
    print(f"final_model_dir: {summary.final_model_dir}")
    print(f"summary: {Path(summary.output_dir) / 'training_summary.json'}")
    logger.info("EmbeddingGemma fine-tuning 완료: final_model_dir=%s", summary.final_model_dir)
    return 0


def _cmd_fine_tune_wav2vec2(
    csv_path: str,
    mp4_dir: str,
    model_name: str,
    output_dir: str,
    epochs: float,
    batch_size: int,
    learning_rate: float,
    warmup_ratio: float,
    eval_fraction: float,
    seed: int,
    device: str,
    fp16: bool,
    bf16: bool,
    max_steps: int | None,
    save_total_limit: int,
    eval_steps: int,
    early_stopping_metric: str,
    early_stopping_patience: int,
    early_stopping_threshold: float,
    sampling_rate: int,
    max_audio_seconds: float | None,
    min_audio_seconds: float | None,
    freeze_feature_encoder: bool,
    on_error: str,
    log_level: str,
    log_file: str | None,
) -> int:
    configure_logging(log_level, log_file)
    from meld_emotion.fine_tunning.wav2vec2 import (
        Wav2Vec2FineTuneConfig,
        run_wav2vec2_fine_tuning,
    )

    logger.info("Wav2Vec2 fine-tuning 준비: csv=%s mp4_dir=%s", csv_path, mp4_dir)
    summary = run_wav2vec2_fine_tuning(
        Wav2Vec2FineTuneConfig(
            csv_path=Path(csv_path),
            mp4_dir=Path(mp4_dir),
            model_name=model_name,
            output_dir=Path(output_dir),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            eval_fraction=eval_fraction,
            seed=seed,
            device=device,
            fp16=fp16,
            bf16=bf16,
            max_steps=max_steps,
            save_total_limit=save_total_limit,
            eval_steps=eval_steps,
            early_stopping_metric=early_stopping_metric,
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
            sampling_rate=sampling_rate,
            max_audio_seconds=max_audio_seconds,
            min_audio_seconds=min_audio_seconds,
            freeze_feature_encoder=freeze_feature_encoder,
            on_error=on_error,
        )
    )
    print(f"final_classifier_dir: {summary.final_classifier_dir}")
    print(f"final_encoder_dir: {summary.final_encoder_dir}")
    print(f"summary: {Path(summary.output_dir) / 'training_summary.json'}")
    logger.info("Wav2Vec2 fine-tuning 완료: final_encoder_dir=%s", summary.final_encoder_dir)
    return 0


def _cmd_fine_tune_timesformer(
    csv_path: str,
    mp4_dir: str,
    model_name: str,
    output_dir: str,
    epochs: float,
    batch_size: int,
    learning_rate: float,
    warmup_ratio: float,
    eval_fraction: float,
    seed: int,
    device: str,
    fp16: bool,
    bf16: bool,
    max_steps: int | None,
    save_total_limit: int,
    eval_steps: int,
    early_stopping_metric: str,
    early_stopping_patience: int,
    early_stopping_threshold: float,
    num_frames: int,
    frame_size: int,
    freeze_backbone: bool,
    on_error: str,
    log_level: str,
    log_file: str | None,
) -> int:
    configure_logging(log_level, log_file)
    from meld_emotion.fine_tunning.timesformer import (
        TimeSformerFineTuneConfig,
        run_timesformer_fine_tuning,
    )

    logger.info("TimeSformer fine-tuning 준비: csv=%s mp4_dir=%s", csv_path, mp4_dir)
    summary = run_timesformer_fine_tuning(
        TimeSformerFineTuneConfig(
            csv_path=Path(csv_path),
            mp4_dir=Path(mp4_dir),
            model_name=model_name,
            output_dir=Path(output_dir),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            eval_fraction=eval_fraction,
            seed=seed,
            device=device,
            fp16=fp16,
            bf16=bf16,
            max_steps=max_steps,
            save_total_limit=save_total_limit,
            eval_steps=eval_steps,
            early_stopping_metric=early_stopping_metric,
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
            num_frames=num_frames,
            frame_size=frame_size,
            freeze_backbone=freeze_backbone,
            on_error=on_error,
        )
    )
    print(f"final_classifier_dir: {summary.final_classifier_dir}")
    print(f"final_encoder_dir: {summary.final_encoder_dir}")
    print(f"summary: {Path(summary.output_dir) / 'training_summary.json'}")
    logger.info("TimeSformer fine-tuning 완료: final_encoder_dir=%s", summary.final_encoder_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meld-emotion", description="MELD 멀티모달 감정 인식")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="YAML 설정으로 실험 실행")
    run_parser.add_argument("--config", required=True, help="실험 설정 YAML 경로")
    _add_logging_args(run_parser)

    compare_parser = sub.add_parser("compare", help="여러 실험을 실행하고 비교표 출력")
    compare_parser.add_argument("--config", required=True, help="비교 묶음(suite) YAML 경로")
    _add_logging_args(compare_parser)

    infer_parser = sub.add_parser("infer", help="MP4 파일과 텍스트로 단일 감정 추론")
    infer_parser.add_argument("--mp4", required=True, help="추론할 MP4 파일 경로")
    infer_parser.add_argument("--text", required=True, help="MP4 발화에 대응하는 텍스트")
    infer_parser.add_argument(
        "--checkpoint",
        default="outputs/best_model.pt",
        help="dialogue_rnn checkpoint 또는 저장된 classifier artifact 경로",
    )
    infer_parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
        help="추론 장치",
    )
    infer_parser.add_argument("--top-k", type=int, default=7, help="출력할 상위 감정 수")
    infer_parser.add_argument(
        "--xai",
        action="store_true",
        help="감정 예측과 함께 fine-grained XAI 결과를 계산",
    )
    infer_parser.add_argument(
        "--xai-steps",
        type=int,
        default=32,
        help="Integrated Gradients step 수",
    )
    infer_parser.add_argument(
        "--xai-top-k",
        type=int,
        default=10,
        help="XAI 항목별/모달리티별 top-k 개수",
    )
    infer_parser.add_argument(
        "--xai-audio-window-seconds",
        type=float,
        default=0.5,
        help="SVM XAI 오디오 ablation 창 길이(초)",
    )
    infer_parser.add_argument(
        "--xai-video-window-seconds",
        type=float,
        default=0.5,
        help="SVM XAI 비디오 ablation 창 길이(초)",
    )
    infer_parser.add_argument(
        "--xai-max-units-per-modality",
        type=int,
        default=0,
        help="SVM XAI 에서 계산할 모달리티별 최대 unit 수(0=제한 없음)",
    )
    infer_parser.add_argument(
        "--xai-dashboard",
        default=None,
        help="단일 inference dashboard JSON 저장 경로",
    )
    infer_parser.add_argument(
        "--markdown-output",
        default=None,
        help="단일 inference 결과와 XAI 설명을 Markdown 으로 저장할 경로",
    )
    infer_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="결과를 JSON 으로 출력",
    )
    _add_logging_args(infer_parser)

    batch_parser = sub.add_parser(
        "infer-batch",
        help="MELD CSV/MP4 디렉터리 전체를 추론하고 XAI 분석 리포트 생성",
    )
    batch_parser.add_argument("--csv", required=True, help="MELD *_sent_emo.csv 경로")
    batch_parser.add_argument("--mp4-dir", required=True, help="MP4 split 디렉터리")
    batch_parser.add_argument(
        "--checkpoint",
        default="outputs/best_model.pt",
        help="dialogue_rnn checkpoint 경로",
    )
    batch_parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
        help="추론 장치",
    )
    batch_parser.add_argument(
        "--predictions",
        default="outputs/test_batch_xai_predictions.jsonl",
        help="샘플별 prediction/XAI JSONL 출력 경로",
    )
    batch_parser.add_argument(
        "--summary",
        default="outputs/test_batch_xai_summary.json",
        help="통합 분석 JSON 출력 경로",
    )
    batch_parser.add_argument(
        "--report",
        default="outputs/dialogue_rnn_xai_analysis.md",
        help="통합 분석 Markdown 리포트 경로",
    )
    batch_parser.add_argument(
        "--manifest",
        default=None,
        help="평가 UID manifest JSON 출력 경로(기본: predictions 옆 eval_manifest.json)",
    )
    batch_parser.add_argument(
        "--suite",
        default="outputs/all_model_w_all_features.json",
        help="SVM/dialogue_rnn 비교 기준 suite JSON 경로",
    )
    batch_parser.add_argument(
        "--xai-steps",
        type=int,
        default=32,
        help="Integrated Gradients step 수",
    )
    batch_parser.add_argument(
        "--xai-top-k",
        type=int,
        default=10,
        help="XAI 항목별 top-k 개수",
    )
    batch_parser.add_argument(
        "--resume",
        action="store_true",
        help="기존 JSONL 의 처리 완료 uid 를 건너뜀",
    )
    batch_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="스모크 테스트용 최대 샘플 수",
    )
    batch_parser.add_argument(
        "--duplicate-uid-policy",
        choices=("fail_fast", "drop_all_rows_with_duplicated_uid"),
        default="fail_fast",
        help="JSONL 중복 uid 처리 정책",
    )
    _add_logging_args(batch_parser)

    svm_batch_parser = sub.add_parser(
        "infer-svm-batch",
        help="MELD CSV/MP4 디렉터리 전체를 저장된 SVM artifact 로 추론하고 XAI 생성",
    )
    svm_batch_parser.add_argument("--csv", required=True, help="MELD *_sent_emo.csv 경로")
    svm_batch_parser.add_argument("--mp4-dir", required=True, help="MP4 split 디렉터리")
    svm_batch_parser.add_argument(
        "--checkpoint",
        required=True,
        help="저장된 SVM/classifier artifact(.pkl) 경로",
    )
    svm_batch_parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
        help="추론 장치",
    )
    svm_batch_parser.add_argument(
        "--predictions",
        default="outputs/svm_test_xai_predictions.jsonl",
        help="샘플별 SVM prediction/XAI JSONL 출력 경로",
    )
    svm_batch_parser.add_argument(
        "--summary",
        default="outputs/svm_test_xai_summary.json",
        help="SVM batch 요약 JSON 출력 경로",
    )
    svm_batch_parser.add_argument("--top-k", type=int, default=7, help="출력할 상위 감정 수")
    svm_batch_parser.add_argument(
        "--xai-top-k",
        type=int,
        default=10,
        help="SVM XAI 모달리티별 top-k unit 개수",
    )
    svm_batch_parser.add_argument(
        "--xai-audio-window-seconds",
        type=float,
        default=0.5,
        help="SVM XAI 오디오 ablation 창 길이(초)",
    )
    svm_batch_parser.add_argument(
        "--xai-video-window-seconds",
        type=float,
        default=0.5,
        help="SVM XAI 비디오 ablation 창 길이(초)",
    )
    svm_batch_parser.add_argument(
        "--xai-max-units-per-modality",
        type=int,
        default=0,
        help="계산할 모달리티별 최대 unit 수(0=제한 없음)",
    )
    svm_batch_parser.add_argument(
        "--resume",
        action="store_true",
        help="기존 JSONL 의 처리 완료 uid 를 건너뜀",
    )
    svm_batch_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="스모크 테스트용 최대 샘플 수",
    )
    _add_logging_args(svm_batch_parser)

    fine_tune_parser = sub.add_parser(
        "fine-tune-embeddinggemma",
        help="MELD 감정 라벨로 EmbeddingGemma sentence embedding fine-tuning",
    )
    fine_tune_parser.add_argument(
        "--csv",
        default="MELD.Raw/train/train_sent_emo.csv",
        help="MELD train_sent_emo.csv 경로",
    )
    fine_tune_parser.add_argument(
        "--model-name",
        default="google/embeddinggemma-300m",
        help="fine-tuning 할 SentenceTransformer 모델 이름 또는 경로",
    )
    fine_tune_parser.add_argument(
        "--output-dir",
        default="outputs/embeddinggemma_meld_finetuned",
        help="checkpoint 와 final 모델을 저장할 디렉터리",
    )
    fine_tune_parser.add_argument("--epochs", type=float, default=1.0, help="학습 epoch 수")
    fine_tune_parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="device 별 train/eval batch size",
    )
    fine_tune_parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="optimizer learning rate",
    )
    fine_tune_parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="전체 step 대비 warmup 비율",
    )
    fine_tune_parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.1,
        help="라벨별 eval split 비율",
    )
    fine_tune_parser.add_argument("--seed", type=int, default=0, help="split/training seed")
    fine_tune_parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
        help="학습 장치",
    )
    fine_tune_parser.add_argument("--fp16", action="store_true", help="fp16 mixed precision 사용")
    fine_tune_parser.add_argument("--bf16", action="store_true", help="bf16 mixed precision 사용")
    fine_tune_parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="smoke run 용 최대 학습 step 수",
    )
    fine_tune_parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="보존할 trainer checkpoint 최대 개수",
    )
    fine_tune_parser.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="validation 평가와 checkpoint 저장 step 간격",
    )
    fine_tune_parser.add_argument(
        "--early-stopping-metric",
        choices=("none", "eval_loss", "eval_macro_f1", "eval_weighted_f1"),
        default="eval_loss",
        help="early stopping 에 사용할 validation 지표",
    )
    fine_tune_parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="개선이 없을 때 기다릴 validation 평가 횟수",
    )
    fine_tune_parser.add_argument(
        "--early-stopping-threshold",
        type=float,
        default=0.0,
        help="개선으로 인정할 최소 지표 변화량",
    )
    _add_logging_args(fine_tune_parser)

    wav2vec2_parser = sub.add_parser(
        "fine-tune-wav2vec2",
        help="MELD MP4 오디오 감정 라벨로 Wav2Vec2 XLS-R fine-tuning",
    )
    wav2vec2_parser.add_argument(
        "--csv",
        default="MELD.Raw/train/train_sent_emo.csv",
        help="MELD train_sent_emo.csv 경로",
    )
    wav2vec2_parser.add_argument(
        "--mp4-dir",
        default="MELD.Raw/train/train_splits",
        help="MELD train split MP4 디렉터리",
    )
    wav2vec2_parser.add_argument(
        "--model-name",
        default="facebook/wav2vec2-xls-r-300m",
        help="fine-tuning 할 Wav2Vec2 모델 이름 또는 경로",
    )
    wav2vec2_parser.add_argument(
        "--output-dir",
        default="outputs/wav2vec2_meld_finetuned",
        help="checkpoint, classifier, encoder 를 저장할 디렉터리",
    )
    wav2vec2_parser.add_argument("--epochs", type=float, default=1.0, help="학습 epoch 수")
    wav2vec2_parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="device 별 train/eval batch size",
    )
    wav2vec2_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="optimizer learning rate",
    )
    wav2vec2_parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="전체 step 대비 warmup 비율",
    )
    wav2vec2_parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.1,
        help="라벨별 eval split 비율",
    )
    wav2vec2_parser.add_argument("--seed", type=int, default=0, help="split/training seed")
    wav2vec2_parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
        help="학습 장치",
    )
    wav2vec2_parser.add_argument("--fp16", action="store_true", help="fp16 mixed precision 사용")
    wav2vec2_parser.add_argument("--bf16", action="store_true", help="bf16 mixed precision 사용")
    wav2vec2_parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="smoke run 용 최대 학습 step 수",
    )
    wav2vec2_parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="보존할 trainer checkpoint 최대 개수",
    )
    wav2vec2_parser.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="validation 평가와 checkpoint 저장 step 간격",
    )
    wav2vec2_parser.add_argument(
        "--early-stopping-metric",
        choices=("none", "eval_loss", "eval_macro_f1", "eval_weighted_f1"),
        default="eval_loss",
        help="early stopping 에 사용할 validation 지표",
    )
    wav2vec2_parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="개선이 없을 때 기다릴 validation 평가 횟수",
    )
    wav2vec2_parser.add_argument(
        "--early-stopping-threshold",
        type=float,
        default=0.0,
        help="개선으로 인정할 최소 지표 변화량",
    )
    wav2vec2_parser.add_argument(
        "--sampling-rate",
        type=int,
        default=16000,
        help="Wav2Vec2 입력 sample rate",
    )
    wav2vec2_parser.add_argument(
        "--max-audio-seconds",
        type=float,
        default=60.0,
        help="drop 할 MP4 오디오 최대 길이",
    )
    wav2vec2_parser.add_argument(
        "--min-audio-seconds",
        type=float,
        default=0.025,
        help="drop 할 MP4 오디오 최소 길이",
    )
    wav2vec2_parser.add_argument(
        "--no-freeze-feature-encoder",
        action="store_false",
        dest="freeze_feature_encoder",
        help="Wav2Vec2 convolution feature encoder 도 함께 fine-tuning",
    )
    wav2vec2_parser.add_argument(
        "--on-error",
        choices=("drop_sample", "fail_fast"),
        default="drop_sample",
        help="누락/손상 MP4 처리 방식",
    )
    _add_logging_args(wav2vec2_parser)

    timesformer_parser = sub.add_parser(
        "fine-tune-timesformer",
        help="MELD MP4 비디오 감정 라벨로 TimeSformer fine-tuning",
    )
    timesformer_parser.add_argument(
        "--csv",
        default="MELD.Raw/train/train_sent_emo.csv",
        help="MELD train_sent_emo.csv 경로",
    )
    timesformer_parser.add_argument(
        "--mp4-dir",
        default="MELD.Raw/train/train_splits",
        help="MELD train split MP4 디렉터리",
    )
    timesformer_parser.add_argument(
        "--model-name",
        default="facebook/timesformer-base-finetuned-k400",
        help="fine-tuning 할 TimeSformer 모델 이름 또는 경로",
    )
    timesformer_parser.add_argument(
        "--output-dir",
        default="outputs/timesformer_meld_finetuned",
        help="checkpoint, classifier, encoder 를 저장할 디렉터리",
    )
    timesformer_parser.add_argument("--epochs", type=float, default=1.0, help="학습 epoch 수")
    timesformer_parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="device 별 train/eval batch size",
    )
    timesformer_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="optimizer learning rate",
    )
    timesformer_parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="전체 step 대비 warmup 비율",
    )
    timesformer_parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.1,
        help="라벨별 eval split 비율",
    )
    timesformer_parser.add_argument("--seed", type=int, default=0, help="split/training seed")
    timesformer_parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
        help="학습 장치",
    )
    timesformer_parser.add_argument("--fp16", action="store_true", help="fp16 mixed precision 사용")
    timesformer_parser.add_argument("--bf16", action="store_true", help="bf16 mixed precision 사용")
    timesformer_parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="smoke run 용 최대 학습 step 수",
    )
    timesformer_parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="보존할 trainer checkpoint 최대 개수",
    )
    timesformer_parser.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="validation 평가와 checkpoint 저장 step 간격",
    )
    timesformer_parser.add_argument(
        "--early-stopping-metric",
        choices=("none", "eval_loss", "eval_macro_f1", "eval_weighted_f1"),
        default="eval_loss",
        help="early stopping 에 사용할 validation 지표",
    )
    timesformer_parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="개선이 없을 때 기다릴 validation 평가 횟수",
    )
    timesformer_parser.add_argument(
        "--early-stopping-threshold",
        type=float,
        default=0.0,
        help="개선으로 인정할 최소 지표 변화량",
    )
    timesformer_parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="TimeSformer 입력으로 샘플링할 프레임 수",
    )
    timesformer_parser.add_argument(
        "--frame-size",
        type=int,
        default=224,
        help="TimeSformer 입력 프레임 한 변 크기",
    )
    timesformer_parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="TimeSformer backbone 을 고정하고 classification head 만 학습",
    )
    timesformer_parser.add_argument(
        "--on-error",
        choices=("drop_sample", "fail_fast"),
        default="drop_sample",
        help="누락/손상 MP4 처리 방식",
    )
    _add_logging_args(timesformer_parser)

    sub.add_parser("status", help="컴포넌트 구현 상태 출력")
    return parser


def _add_logging_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="진행 상황 로그 레벨",
    )
    parser.add_argument("--log-file", default=None, help="로그를 추가로 저장할 파일 경로")


def main(argv: Sequence[str] | None = None) -> int:
    _force_utf8()
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return _cmd_run(args.config, args.log_level, args.log_file)
    if args.command == "compare":
        return _cmd_compare(args.config, args.log_level, args.log_file)
    if args.command == "infer":
        return _cmd_infer(
            args.mp4,
            args.text,
            args.checkpoint,
            args.device,
            args.top_k,
            args.json_output,
            args.xai,
            args.xai_steps,
            args.xai_top_k,
            args.xai_audio_window_seconds,
            args.xai_video_window_seconds,
            args.xai_max_units_per_modality,
            args.xai_dashboard,
            args.markdown_output,
            args.log_level,
            args.log_file,
        )
    if args.command == "infer-batch":
        return _cmd_infer_batch(
            args.csv,
            args.mp4_dir,
            args.checkpoint,
            args.device,
            args.predictions,
            args.summary,
            args.report,
            args.manifest,
            args.suite,
            args.xai_steps,
            args.xai_top_k,
            args.resume,
            args.limit,
            args.duplicate_uid_policy,
            args.log_level,
            args.log_file,
        )
    if args.command == "infer-svm-batch":
        return _cmd_infer_svm_batch(
            args.csv,
            args.mp4_dir,
            args.checkpoint,
            args.device,
            args.predictions,
            args.summary,
            args.top_k,
            args.xai_top_k,
            args.xai_audio_window_seconds,
            args.xai_video_window_seconds,
            args.xai_max_units_per_modality,
            args.resume,
            args.limit,
            args.log_level,
            args.log_file,
        )
    if args.command == "fine-tune-embeddinggemma":
        return _cmd_fine_tune_embeddinggemma(
            args.csv,
            args.model_name,
            args.output_dir,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.warmup_ratio,
            args.eval_fraction,
            args.seed,
            args.device,
            args.fp16,
            args.bf16,
            args.max_steps,
            args.save_total_limit,
            args.eval_steps,
            args.early_stopping_metric,
            args.early_stopping_patience,
            args.early_stopping_threshold,
            args.log_level,
            args.log_file,
        )
    if args.command == "fine-tune-wav2vec2":
        return _cmd_fine_tune_wav2vec2(
            args.csv,
            args.mp4_dir,
            args.model_name,
            args.output_dir,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.warmup_ratio,
            args.eval_fraction,
            args.seed,
            args.device,
            args.fp16,
            args.bf16,
            args.max_steps,
            args.save_total_limit,
            args.eval_steps,
            args.early_stopping_metric,
            args.early_stopping_patience,
            args.early_stopping_threshold,
            args.sampling_rate,
            args.max_audio_seconds,
            args.min_audio_seconds,
            args.freeze_feature_encoder,
            args.on_error,
            args.log_level,
            args.log_file,
        )
    if args.command == "fine-tune-timesformer":
        return _cmd_fine_tune_timesformer(
            args.csv,
            args.mp4_dir,
            args.model_name,
            args.output_dir,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.warmup_ratio,
            args.eval_fraction,
            args.seed,
            args.device,
            args.fp16,
            args.bf16,
            args.max_steps,
            args.save_total_limit,
            args.eval_steps,
            args.early_stopping_metric,
            args.early_stopping_patience,
            args.early_stopping_threshold,
            args.num_frames,
            args.frame_size,
            args.freeze_backbone,
            args.on_error,
            args.log_level,
            args.log_file,
        )
    if args.command == "status":
        return _cmd_status()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
