import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import soundfile as sf


def hz_to_bin(hz: float, n_fft: int, sr: int) -> int:
    return int(np.round(hz / sr * n_fft))


def merge_intervals(intervals: List[Tuple[float, float]], gap_s: float) -> List[Tuple[float, float]]:
    """Merge intervals whose gap <= gap_s (seconds)."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s - merged[-1][1] <= gap_s:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(float(s), float(e)) for s, e in merged]


def invert_intervals(
    intervals: List[Tuple[float, float]],
    total_duration_s: float,
    gap_s: float = 0.0
) -> List[Tuple[float, float]]:
    """将语音段取补集得到非语音段。"""
    if total_duration_s <= 0:
        return []
    merged = merge_intervals(intervals, gap_s=gap_s)
    non_speech: List[Tuple[float, float]] = []
    cur = 0.0
    for s, e in merged:
        s = max(0.0, min(float(s), total_duration_s))
        e = max(0.0, min(float(e), total_duration_s))
        if s > cur:
            non_speech.append((cur, s))
        cur = max(cur, e)
    if cur < total_duration_s:
        non_speech.append((cur, total_duration_s))
    return non_speech


def _load_silero_vad(vad_repo: Optional[str]):
    """按需从本地仓库或已安装包加载 silero_vad。"""
    try:
        import silero_vad  # type: ignore
        return silero_vad
    except Exception:
        if not vad_repo:
            raise
        # 允许从本地仓库目录加载
        repo_path = os.path.abspath(vad_repo)
        repo_src = os.path.join(repo_path, "src")
        if os.path.isdir(repo_src) and repo_src not in sys.path:
            sys.path.insert(0, repo_src)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        import silero_vad  # type: ignore
        return silero_vad


@dataclass
class BreathRuleConfig:
    # Framing
    frame_ms: float = 20.0
    hop_ms: float = 10.0

    # Segment post-processing
    min_cont_ms: float = 80.0
    merge_gap_ms: float = 120.0

    # Spectral ratio band (Hz)
    ratio_low_hz: float = 1000.0
    ratio_high_hz: float = 8000.0

    # Thresholds
    zcr_min: float = 0.03
    band_ratio_min: float = 0.10
    energy_db_above_noise: float = 6.0

    # Robust stats
    noise_percentile: float = 5.0    # for noise floor
    speech_percentile: float = 99.0  # for speech ref (upper energy)

    # Speech+breath mode (upper bound)
    use_upper_bound: bool = True
    margin_db: float = 10.0          # breaths expected below speech_ref by this margin
    min_band_db: float = 8.0         # ensure e_max >= e_min + min_band_db

    # Debug printing
    debug: bool = True


@dataclass
class VadConfig:
    # silero-vad 相关参数
    sampling_rate: int = 16000
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    max_speech_duration_s: float = float("inf")
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 30
    return_seconds: bool = True
    time_resolution: int = 3
    window_size_samples: int = 512
    onnx: bool = False
    vad_repo: Optional[str] = None


def compute_features(x: np.ndarray, sr: int, cfg: BreathRuleConfig) -> Dict[str, np.ndarray]:
    """Compute frame-wise energy(dB), ZCR, and band energy ratio."""
    x = x.astype(np.float32)
    x = x / (np.max(np.abs(x)) + 1e-9)

    frame_len = int(sr * cfg.frame_ms / 1000.0)
    hop_len = int(sr * cfg.hop_ms / 1000.0)

    n_fft = 1
    while n_fft < frame_len:
        n_fft *= 2

    window = np.hanning(frame_len).astype(np.float32)

    n_frames = 1 + max(0, (len(x) - frame_len) // hop_len)
    if n_frames <= 0:
        raise ValueError("Audio too short for the given frame_ms/hop_ms.")

    energies = np.zeros(n_frames, dtype=np.float32)
    zcrs = np.zeros(n_frames, dtype=np.float32)
    ratios = np.zeros(n_frames, dtype=np.float32)

    # freq bins for ratio: E(ratio_low~ratio_high)/E(0~ratio_high)
    hi_hz = min(cfg.ratio_high_hz, sr / 2)
    lo_bin = hz_to_bin(0.0, n_fft, sr)
    hi_bin = hz_to_bin(hi_hz, n_fft, sr)
    b1_bin = hz_to_bin(cfg.ratio_low_hz, n_fft, sr)

    eps = 1e-12
    for i in range(n_frames):
        s = i * hop_len
        frame = x[s:s + frame_len]
        if len(frame) < frame_len:
            # pad last frame if needed
            pad = np.zeros(frame_len - len(frame), dtype=np.float32)
            frame = np.concatenate([frame, pad])
        frame = frame * window

        # short-time energy
        e = float(np.mean(frame * frame) + eps)
        energies[i] = e

        # ZCR
        signs = np.sign(frame)
        signs[signs == 0] = 1
        zcrs[i] = 0.5 * np.mean(np.abs(np.diff(signs)))

        # band energy ratio
        spec = np.fft.rfft(frame, n=n_fft)
        mag2 = (np.abs(spec) ** 2).astype(np.float32)
        e_total = float(np.sum(mag2[lo_bin:hi_bin + 1]) + eps)
        e_band = float(np.sum(mag2[b1_bin:hi_bin + 1]) + eps)
        ratios[i] = e_band / e_total

    edb = 10.0 * np.log10(energies + eps).astype(np.float32)

    return {
        "edb": edb,
        "zcr": zcrs,
        "ratio": ratios,
        "frame_len": frame_len,
        "hop_len": hop_len,
    }


def decide_thresholds(edb: np.ndarray, cfg: BreathRuleConfig) -> Dict[str, float]:
    """Compute adaptive energy thresholds with guaranteed valid range."""
    noise_floor = float(np.percentile(edb, cfg.noise_percentile))
    speech_ref = float(np.percentile(edb, cfg.speech_percentile))

    e_min = noise_floor + cfg.energy_db_above_noise

    if cfg.use_upper_bound:
        # Clamp: ensure range exists
        e_max = max(speech_ref - cfg.margin_db, e_min + cfg.min_band_db)
    else:
        e_max = float("inf")

    return {
        "noise_floor_db": noise_floor,
        "speech_ref_db": speech_ref,
        "energy_db_min": float(e_min),
        "energy_db_max": float(e_max if np.isfinite(e_max) else 1e9),
    }


def frames_to_segments(
    is_pos: np.ndarray,
    sr: int,
    frame_len: int,
    hop_len: int,
    cfg: BreathRuleConfig
) -> List[Tuple[float, float]]:
    """Convert boolean mask to time segments with min continuous duration and merging."""
    min_cont_frames = int(np.ceil((cfg.min_cont_ms / 1000.0) / (hop_len / sr)))
    segments: List[Tuple[float, float]] = []

    i = 0
    n_frames = len(is_pos)
    while i < n_frames:
        if is_pos[i]:
            j = i
            while j < n_frames and is_pos[j]:
                j += 1
            if (j - i) >= min_cont_frames:
                start_t = (i * hop_len) / sr
                end_t = ((j * hop_len) + frame_len) / sr
                segments.append((float(start_t), float(end_t)))
            i = j
        else:
            i += 1

    segments = merge_intervals(segments, gap_s=cfg.merge_gap_ms / 1000.0)
    return segments


def detect_breath_rule_from_array(x: np.ndarray, sr: int, cfg: BreathRuleConfig) -> Dict[str, Any]:
    """对给定音频数组做规则法呼吸检测。"""
    if x.ndim > 1:
        x = np.mean(x, axis=1)

    feats = compute_features(x, sr, cfg)
    edb = feats["edb"]
    zcr = feats["zcr"]
    ratio = feats["ratio"]
    frame_len = feats["frame_len"]
    hop_len = feats["hop_len"]

    thr = decide_thresholds(edb, cfg)
    e_min = thr["energy_db_min"]
    e_max = thr["energy_db_max"]

    # 帧级判定
    if np.isfinite(e_max):
        is_breath = (edb >= e_min) & (edb <= e_max) & (zcr >= cfg.zcr_min) & (ratio >= cfg.band_ratio_min)
    else:
        is_breath = (edb >= e_min) & (zcr >= cfg.zcr_min) & (ratio >= cfg.band_ratio_min)

    segments = frames_to_segments(is_breath, sr, frame_len, hop_len, cfg)

    total_dur = len(x) / sr
    covered = 0.0
    for s, e in segments:
        covered += max(0.0, min(e, total_dur) - max(s, 0.0))
    coverage = covered / max(total_dur, 1e-9)

    if cfg.debug:
        print("=== Debug Stats ===")
        print(f"sr={sr}, dur={total_dur:.3f}s, frames={len(edb)}, frame={cfg.frame_ms}ms hop={cfg.hop_ms}ms")
        print(f"noise_floor={thr['noise_floor_db']:.2f} dB, speech_ref={thr['speech_ref_db']:.2f} dB")
        print(f"e_min={e_min:.2f} dB, e_max={e_max:.2f} dB, use_upper_bound={cfg.use_upper_bound}")
        print(f"edb range:   [{edb.min():.2f}, {edb.max():.2f}]")
        print(f"zcr range:   [{zcr.min():.3f}, {zcr.max():.3f}] (zcr_min={cfg.zcr_min})")
        print(f"ratio range: [{ratio.min():.3f}, {ratio.max():.3f}] (ratio_min={cfg.band_ratio_min})")
        print(f"segments={len(segments)}, coverage={coverage:.3f}")
        print("===================")

    return {
        "config": asdict(cfg),
        "sr": int(sr),
        "duration_s": float(total_dur),
        **thr,
        "coverage": float(coverage),
        "num_segments": int(len(segments)),
        "segments": segments[:500],
    }


def detect_breath_rule(wav_path: str, cfg: BreathRuleConfig) -> Dict[str, Any]:
    x, sr = sf.read(wav_path)
    return detect_breath_rule_from_array(x, sr, cfg)


def detect_breath_on_segments(
    x: np.ndarray,
    sr: int,
    segments: List[Tuple[float, float]],
    cfg: BreathRuleConfig
) -> Dict[str, Any]:
    """仅在指定的时间段内进行呼吸检测，并映射回原始时间轴。"""
    breath_segments: List[Tuple[float, float]] = []
    segment_reports: List[Dict[str, Any]] = []
    for s, e in segments:
        start_idx = int(round(s * sr))
        end_idx = int(round(e * sr))
        if end_idx <= start_idx:
            continue
        seg_x = x[start_idx:end_idx]
        try:
            seg_out = detect_breath_rule_from_array(seg_x, sr, cfg)
        except ValueError:
            # 段太短时直接跳过
            continue
        mapped = [(bs + s, be + s) for bs, be in seg_out["segments"]]
        breath_segments.extend(mapped)
        segment_reports.append({
            "segment": (float(s), float(e)),
            "breath_segments": mapped,
            "coverage": float(seg_out["coverage"]),
            "num_segments": int(seg_out["num_segments"])
        })

    return {
        "breath_segments": breath_segments[:2000],
        "segment_reports": segment_reports[:500]
    }


def detect_breath_with_vad(
    wav_path: str,
    cfg: BreathRuleConfig,
    vad_cfg: VadConfig
) -> Dict[str, Any]:
    """VAD 前置：取非语音段后做呼吸检测。"""
    x, sr = sf.read(wav_path)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    total_dur = len(x) / sr

    silero = _load_silero_vad(vad_cfg.vad_repo)
    model = silero.load_silero_vad(onnx=vad_cfg.onnx)
    wav = silero.read_audio(wav_path, sampling_rate=vad_cfg.sampling_rate)
    speech_timestamps = silero.get_speech_timestamps(
        wav,
        model,
        threshold=vad_cfg.threshold,
        sampling_rate=vad_cfg.sampling_rate,
        min_speech_duration_ms=vad_cfg.min_speech_duration_ms,
        max_speech_duration_s=vad_cfg.max_speech_duration_s,
        min_silence_duration_ms=vad_cfg.min_silence_duration_ms,
        speech_pad_ms=vad_cfg.speech_pad_ms,
        return_seconds=vad_cfg.return_seconds,
        time_resolution=vad_cfg.time_resolution,
        window_size_samples=vad_cfg.window_size_samples
    )
    speech_segments = [(float(s["start"]), float(s["end"])) for s in speech_timestamps]
    non_speech_segments = invert_intervals(speech_segments, total_dur, gap_s=0.0)

    breath_out = detect_breath_on_segments(x, sr, non_speech_segments, cfg)
    breath_segments = breath_out["breath_segments"]

    non_speech_total = sum(max(0.0, e - s) for s, e in non_speech_segments)
    covered_total = sum(max(0.0, min(e, total_dur) - max(s, 0.0)) for s, e in breath_segments)
    coverage_total = covered_total / max(total_dur, 1e-9)
    coverage_non_speech = covered_total / max(non_speech_total, 1e-9)

    return {
        "config": asdict(cfg),
        "vad_config": asdict(vad_cfg),
        "sr": int(sr),
        "duration_s": float(total_dur),
        "speech_segments": speech_segments[:500],
        "non_speech_segments": non_speech_segments[:500],
        "breath_segments": breath_segments[:2000],
        "coverage_total": float(coverage_total),
        "coverage_non_speech": float(coverage_non_speech),
        "num_breath_segments": int(len(breath_segments)),
        "segment_reports": breath_out["segment_reports"]
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", help="path to audio (wav/flac/etc.)")
    ap.add_argument("--mode", choices=["breath_only", "mixed"], default="mixed",
                    help="breath_only disables energy upper bound; mixed enables it (speech+breath).")
    ap.add_argument("--debug", action="store_true", help="print debug stats")
    ap.add_argument("--use-vad", action="store_true", default=True, help="使用 VAD 预处理")
    ap.add_argument("--no-vad", dest="use_vad", action="store_false", help="禁用 VAD 预处理")
    ap.add_argument("--vad-repo", default=None, help="silero-vad 本地仓库路径（未安装包时使用）")
    ap.add_argument("--vad-threshold", type=float, default=0.5, help="VAD 阈值")
    ap.add_argument("--vad-min-speech-ms", type=int, default=250, help="最小语音时长(ms)")
    ap.add_argument("--vad-min-silence-ms", type=int, default=100, help="最小静音时长(ms)")
    ap.add_argument("--vad-speech-pad-ms", type=int, default=30, help="语音段两侧 padding(ms)")
    ap.add_argument("--vad-max-speech-s", type=float, default=float("inf"), help="最大语音段时长(s)")
    ap.add_argument("--vad-sr", type=int, default=16000, help="VAD 采样率")
    ap.add_argument("--vad-onnx", action="store_true", help="使用 onnx 版本 VAD")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ==== Centralized parameters (edit here) ====
    CONFIG = BreathRuleConfig(
        # framing
        frame_ms=20.0,
        hop_ms=10.0,

        # segment
        min_cont_ms=90.0,
        merge_gap_ms=100.0,

        # thresholds
        zcr_min=0.03,
        band_ratio_min=0.22,
        energy_db_above_noise=4.0,

        # robust stats
        noise_percentile=5.0,
        speech_percentile=99.0,

        # energy upper bound behavior
        use_upper_bound=True,   # will be overridden by --mode
        margin_db=9.0,
        min_band_db=10.0,

        debug=args.debug,
    )

    if args.mode == "breath_only":
        CONFIG.use_upper_bound = False
    else:
        CONFIG.use_upper_bound = True

    VAD_CONFIG = VadConfig(
        sampling_rate=args.vad_sr,
        threshold=args.vad_threshold,
        min_speech_duration_ms=args.vad_min_speech_ms,
        max_speech_duration_s=args.vad_max_speech_s,
        min_silence_duration_ms=args.vad_min_silence_ms,
        speech_pad_ms=args.vad_speech_pad_ms,
        return_seconds=True,
        time_resolution=3,
        window_size_samples=512,
        onnx=args.vad_onnx,
        vad_repo=args.vad_repo
    )

    if args.use_vad:
        out = detect_breath_with_vad(args.wav, CONFIG, VAD_CONFIG)
    else:
        out = detect_breath_rule(args.wav, CONFIG)
    print(json.dumps(out, ensure_ascii=False, indent=2))
