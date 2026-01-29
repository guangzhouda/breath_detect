import math
import os
import sys
import numpy as np
import soundfile as sf
import scipy.signal as sps
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

try:
    import webrtcvad  # type: ignore
except Exception:
    webrtcvad = None


# -----------------------------
# Common utilities
# -----------------------------
def read_audio_mono(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    if sr != target_sr:
        # 使用 scipy 重采样，避免额外依赖（librosa）
        g = math.gcd(int(sr), int(target_sr))
        up = int(target_sr) // g
        down = int(sr) // g
        x = sps.resample_poly(x.astype(np.float32), up=up, down=down).astype(np.float32)
        sr = target_sr
    x = x.astype(np.float32)
    # remove DC
    x = x - np.mean(x)
    return x, sr


def _load_silero_vad(vad_repo: Optional[str]):
    """按需从本地仓库或已安装包加载 silero_vad。"""
    try:
        import silero_vad  # type: ignore
        return silero_vad
    except Exception:
        if not vad_repo:
            raise
        repo_path = os.path.abspath(vad_repo)
        repo_src = os.path.join(repo_path, "src")
        if os.path.isdir(repo_src) and repo_src not in sys.path:
            sys.path.insert(0, repo_src)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        import silero_vad  # type: ignore
        return silero_vad


def frame_signal(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))
    n_frames = 1 + (len(x) - frame_len) // hop
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, frame_len),
        strides=(x.strides[0] * hop, x.strides[0]),
        writeable=False
    )
    return frames.copy()


def merge_intervals(intervals: List[Tuple[int, int]], gap: int) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe + gap:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def drop_short(intervals: List[Tuple[int, int]], min_len: int) -> List[Tuple[int, int]]:
    return [(s, e) for s, e in intervals if (e - s) >= min_len]


def intersect_intervals(a: List[Tuple[int, int]], b: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """区间求交（输入为采样点区间）。"""
    if not a or not b:
        return []
    a = merge_intervals(sorted(a), gap=0)
    b = merge_intervals(sorted(b), gap=0)

    out: List[Tuple[int, int]] = []
    i = 0
    j = 0
    while i < len(a) and j < len(b):
        s1, e1 = a[i]
        s2, e2 = b[j]
        s = max(s1, s2)
        e = min(e1, e2)
        if e > s:
            out.append((s, e))
        if e1 <= e2:
            i += 1
        else:
            j += 1
    return out


def intervals_to_mask(n: int, intervals: List[Tuple[int, int]]) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for s, e in intervals:
        s = max(0, s)
        e = min(n, e)
        m[s:e] = True
    return m


def smooth_bool(x: np.ndarray, win: int) -> np.ndarray:
    # majority vote smoothing
    if win <= 1:
        return x
    k = np.ones(win, dtype=np.float32)
    y = np.convolve(x.astype(np.float32), k, mode="same") / win
    return y > 0.5


def _fill_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """将 True 片段之间的短 False 间隙填平，用于减少“一个呼吸被切成多个段”的情况。"""
    if max_gap <= 0:
        return mask
    m = mask.copy()
    n = len(m)
    i = 0
    while i < n:
        if m[i]:
            i += 1
            continue
        # 统计 False 段长度
        j = i
        while j < n and not m[j]:
            j += 1
        gap_len = j - i
        if gap_len <= max_gap:
            left_true = i - 1 >= 0 and m[i - 1]
            right_true = j < n and m[j]
            if left_true and right_true:
                m[i:j] = True
        i = j
    return m


# -----------------------------
# Spectral features for breath (no ZFF)
# -----------------------------
@dataclass
class BreathFeatCfg:
    sr: int = 16000
    frame_ms: float = 20.0
    hop_ms: float = 10.0

    # band energy ratio: (1-3.5k) / (0-1k)
    mid_band: Tuple[int, int] = (1000, 3500)
    low_band: Tuple[int, int] = (0, 1000)

    # breath-ish thresholds (tunable)
    min_rms: float = 1e-4          # too quiet -> ignore
    min_mid_low_ratio: float = 1.2 # ER-like (mid/low)
    min_flatness: float = 0.25     # spectral flatness
    max_spectral_var: float = 6.0  # spectral variance proxy (log-mag variance)


def spectral_breath_features(frames: np.ndarray, cfg: BreathFeatCfg) -> Dict[str, np.ndarray]:
    # STFT per frame with rfft
    win = np.hanning(frames.shape[1]).astype(np.float32)
    X = np.fft.rfft(frames * win[None, :], axis=1)
    mag = np.maximum(np.abs(X), 1e-12)

    # RMS
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)

    # freq bins
    nfft = frames.shape[1]
    freqs = np.fft.rfftfreq(nfft, d=1.0 / cfg.sr)

    def band_energy(f_lo, f_hi):
        idx = np.where((freqs >= f_lo) & (freqs < f_hi))[0]
        if len(idx) == 0:
            return np.zeros(mag.shape[0], dtype=np.float32)
        return np.mean(mag[:, idx]**2, axis=1)

    e_mid = band_energy(*cfg.mid_band)
    e_low = band_energy(*cfg.low_band)
    mid_low_ratio = (e_mid + 1e-12) / (e_low + 1e-12)

    # spectral flatness: geometric mean / arithmetic mean
    geo = np.exp(np.mean(np.log(mag), axis=1))
    ari = np.mean(mag, axis=1) + 1e-12
    flatness = geo / ari

    # spectral variance proxy: variance of log-mag
    logmag = np.log(mag)
    spec_var = np.var(logmag, axis=1)

    return {
        "rms": rms,
        "mid_low_ratio": mid_low_ratio,
        "flatness": flatness,
        "spec_var": spec_var,
    }


def spectral_breath_score(frames: np.ndarray, cfg: BreathFeatCfg) -> Tuple[np.ndarray, dict]:
    aux = spectral_breath_features(frames, cfg)
    rms = aux["rms"]
    mid_low_ratio = aux["mid_low_ratio"]
    flatness = aux["flatness"]
    spec_var = aux["spec_var"]

    # breath candidate rule (frame-level)
    cand = (
        (rms >= cfg.min_rms) &
        (mid_low_ratio >= cfg.min_mid_low_ratio) &
        (flatness >= cfg.min_flatness) &
        (spec_var <= cfg.max_spectral_var)
    )
    return cand, aux


def spectral_breath_cand_from_feats(
    feats: Dict[str, np.ndarray],
    cfg: BreathFeatCfg,
    *,
    min_rms: float,
    min_mid_low_ratio: float
) -> np.ndarray:
    """从已计算好的谱特征生成帧级候选 mask（便于做自适应阈值/回退策略）。"""
    rms = feats["rms"]
    mid_low_ratio = feats["mid_low_ratio"]
    flatness = feats["flatness"]
    spec_var = feats["spec_var"]
    return (
        (rms >= min_rms) &
        (mid_low_ratio >= min_mid_low_ratio) &
        (flatness >= cfg.min_flatness) &
        (spec_var <= cfg.max_spectral_var)
    )


# -----------------------------
# Version A: VAD + no ZFF
# -----------------------------
@dataclass
class VADCfg:
    sr: int = 16000
    frame_ms: int = 30              # webrtcvad supports 10/20/30
    aggressiveness: int = 2         # 0..3
    window_mode: str = "non_speech" # boundary | non_speech
    # breath search window around speech boundaries (seconds)
    boundary_win_s: float = 0.5
    # min breath segment duration
    min_breath_ms: int = 60
    # merge close breath segments
    merge_gap_ms: int = 40


@dataclass
class SileroVadCfg:
    sr: int = 16000
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    max_speech_duration_s: float = float("inf")
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 30
    time_resolution: int = 3
    window_size_samples: int = 512
    onnx: bool = False
    vad_repo: Optional[str] = None


def webrtcvad_speech_mask(x: np.ndarray, cfg: VADCfg) -> np.ndarray:
    if webrtcvad is None:
        raise RuntimeError("webrtcvad 未安装，无法使用 webrtcvad 后端。请改用 silero-vad 或安装 webrtcvad。")
    vad = webrtcvad.Vad(cfg.aggressiveness)
    frame_len = int(cfg.sr * cfg.frame_ms / 1000)
    hop = frame_len  # webrtcvad expects contiguous frames
    # pad to full frames
    pad = (-len(x)) % frame_len
    if pad:
        x = np.pad(x, (0, pad))
    frames = frame_signal(x, frame_len, hop)

    speech = np.zeros(frames.shape[0], dtype=bool)
    for i in range(frames.shape[0]):
        # webrtcvad expects 16-bit PCM bytes
        pcm16 = np.clip(frames[i] * 32768.0, -32768, 32767).astype(np.int16).tobytes()
        speech[i] = vad.is_speech(pcm16, sample_rate=cfg.sr)

    # expand frame-level to sample-level mask
    mask = np.zeros(len(x), dtype=bool)
    for i, s in enumerate(speech):
        if s:
            mask[i*frame_len:(i+1)*frame_len] = True
    return mask[:len(x)]


def extract_speech_intervals(mask: np.ndarray) -> List[Tuple[int, int]]:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    # find contiguous regions
    cuts = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[cuts + 1]]
    ends = np.r_[idx[cuts], idx[-1]]
    return [(int(s), int(e) + 1) for s, e in zip(starts, ends)]


def silero_speech_intervals(x: np.ndarray, cfg: SileroVadCfg) -> List[Tuple[int, int]]:
    """用 silero-vad 得到语音段（返回采样点区间）。"""
    silero = _load_silero_vad(cfg.vad_repo)
    model = silero.load_silero_vad(onnx=cfg.onnx)

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"未安装 torch，无法使用 silero-vad: {exc}") from exc

    wav = torch.from_numpy(x.astype(np.float32, copy=False))
    speech_timestamps = silero.get_speech_timestamps(
        wav,
        model,
        threshold=cfg.threshold,
        sampling_rate=cfg.sr,
        min_speech_duration_ms=cfg.min_speech_duration_ms,
        max_speech_duration_s=cfg.max_speech_duration_s,
        min_silence_duration_ms=cfg.min_silence_duration_ms,
        speech_pad_ms=cfg.speech_pad_ms,
        return_seconds=True,
        time_resolution=cfg.time_resolution,
        window_size_samples=cfg.window_size_samples
    )

    intervals: List[Tuple[int, int]] = []
    for seg in speech_timestamps:
        s = int(round(float(seg["start"]) * cfg.sr))
        e = int(round(float(seg["end"]) * cfg.sr))
        s = max(0, s)
        e = min(len(x), e)
        if e > s:
            intervals.append((s, e))
    return merge_intervals(intervals, gap=0)


def breath_detect_vad_no_zff(
    x: np.ndarray,
    vad_cfg: VADCfg,
    feat_cfg: BreathFeatCfg,
    speech_intervals: Optional[List[Tuple[int, int]]] = None,
    auto_tune: bool = False,
    auto_ratio_percentile: float = 90.0,
    auto_ratio_min_percentile: float = 80.0,
    auto_backoff_step: float = 5.0,
    auto_rms_percentile: float = 20.0,
    auto_rms_factor: float = 1.2,
    debug: bool = False
) -> List[Tuple[int, int]]:
    if speech_intervals is None:
        speech_mask = webrtcvad_speech_mask(x, vad_cfg)
        speech_intervals = extract_speech_intervals(speech_mask)

    # 预先算出非语音段（语音段的补集），用于约束搜索范围
    windows: List[Tuple[int, int]] = []
    speech_intervals = merge_intervals(speech_intervals, gap=int(0.02 * vad_cfg.sr))

    non_speech: List[Tuple[int, int]] = []
    if len(speech_intervals) == 0:
        non_speech = [(0, len(x))]
    else:
        cur = 0
        for s, e in speech_intervals:
            if s > cur:
                non_speech.append((cur, s))
            cur = max(cur, e)
        if cur < len(x):
            non_speech.append((cur, len(x)))

    if vad_cfg.window_mode == "boundary" and len(speech_intervals) > 0:
        # 只在语音边界附近搜索，但仍强制只取非语音部分，避免窗口跨进下一段语音
        win = int(vad_cfg.boundary_win_s * vad_cfg.sr)
        boundary_windows: List[Tuple[int, int]] = []
        for s, e in speech_intervals:
            boundary_windows.append((max(0, s - win), s))              # before start
            boundary_windows.append((e, min(len(x), e + win)))         # after end
        boundary_windows = merge_intervals(boundary_windows, gap=int(0.05 * vad_cfg.sr))
        windows = intersect_intervals(boundary_windows, non_speech)
    else:
        # 默认：全非语音段搜索
        windows = non_speech

    # Frame-level breath detection inside windows only
    frame_len = int(feat_cfg.sr * feat_cfg.frame_ms / 1000)
    hop = int(feat_cfg.sr * feat_cfg.hop_ms / 1000)

    # 预先在所有搜索窗口内计算特征（避免 auto 回退时重复做 FFT）
    win_cache: List[Tuple[int, int, Dict[str, np.ndarray]]] = []
    ratio_all: List[np.ndarray] = []
    rms_all: List[np.ndarray] = []
    for ws, we in windows:
        seg = x[ws:we]
        if len(seg) < frame_len:
            continue
        frames = frame_signal(seg, frame_len, hop)
        feats = spectral_breath_features(frames, feat_cfg)
        win_cache.append((ws, we, feats))
        ratio_all.append(feats["mid_low_ratio"])
        rms_all.append(feats["rms"])

    # --- auto：在当前音频上做“只放宽不收紧”的自适应阈值，并带回退（每段 non_speech 单独兜底） ---
    ratio_thr_base = float(feat_cfg.min_mid_low_ratio)
    min_rms_dyn = float(feat_cfg.min_rms)
    if auto_tune and ratio_all:
        ratio_cat = np.concatenate(ratio_all)
        p = float(np.percentile(ratio_cat, auto_ratio_percentile))
        ratio_thr_base = float(min(feat_cfg.min_mid_low_ratio, p))
        if debug:
            print(f"[auto] mid_low_ratio p{auto_ratio_percentile:.0f}={p:.3f} -> thr_base={ratio_thr_base:.3f} (cap={feat_cfg.min_mid_low_ratio})")

    if auto_tune and rms_all:
        rms_cat = np.concatenate(rms_all)
        noise_rms = float(np.percentile(rms_cat, auto_rms_percentile))
        min_rms_dyn = float(max(feat_cfg.min_rms, noise_rms * auto_rms_factor))
        if debug:
            print(f"[auto] rms noise(p{auto_rms_percentile:.0f})={noise_rms:.6f} -> min_rms={min_rms_dyn:.6f} (factor={auto_rms_factor})")

    breath_intervals = []
    for ws, we, feats in win_cache:
        # 先用全局阈值检测
        cand = spectral_breath_cand_from_feats(
            feats,
            feat_cfg,
            min_rms=min_rms_dyn,
            min_mid_low_ratio=ratio_thr_base
        )

        # 填平短间隙，减少“一个呼吸被切成多个段”
        gap_frames = int(round((vad_cfg.merge_gap_ms / 1000.0) * feat_cfg.sr / hop))
        cand = _fill_short_gaps(cand, max_gap=max(0, gap_frames))

        # convert frame cand -> sample intervals in seg
        cand_s = []
        for i, c in enumerate(cand):
            if c:
                ss = i * hop
                ee = ss + frame_len
                cand_s.append((ss, ee))
        cand_s = merge_intervals(cand_s, gap=int(vad_cfg.merge_gap_ms * feat_cfg.sr / 1000))
        cand_s = drop_short(cand_s, min_len=int(vad_cfg.min_breath_ms * feat_cfg.sr / 1000))

        # auto 回退：如果这个 non_speech 段完全没有结果，但段内存在明显能量峰值，则单段放宽 ratio 再试一次
        if auto_tune and not cand_s:
            peak_rms = float(np.max(feats["rms"])) if len(feats["rms"]) else 0.0
            # 没有“明显声音”的段直接跳过，避免在纯静音里为了“凑段”而误检
            if peak_rms >= min_rms_dyn * 1.1:
                p0 = float(auto_ratio_percentile)
                p_min = float(auto_ratio_min_percentile)
                step = float(max(1.0, auto_backoff_step))
                p_try = p0
                while p_try - step >= p_min and not cand_s:
                    p_try -= step
                    local_thr = float(min(feat_cfg.min_mid_low_ratio, np.percentile(feats["mid_low_ratio"], p_try)))
                    cand2 = spectral_breath_cand_from_feats(
                        feats,
                        feat_cfg,
                        min_rms=min_rms_dyn,
                        min_mid_low_ratio=local_thr
                    )
                    cand2 = _fill_short_gaps(cand2, max_gap=max(0, gap_frames))
                    cand_s2 = []
                    for i, c in enumerate(cand2):
                        if c:
                            ss = i * hop
                            ee = ss + frame_len
                            cand_s2.append((ss, ee))
                    cand_s2 = merge_intervals(cand_s2, gap=int(vad_cfg.merge_gap_ms * feat_cfg.sr / 1000))
                    cand_s2 = drop_short(cand_s2, min_len=int(vad_cfg.min_breath_ms * feat_cfg.sr / 1000))
                    if cand_s2:
                        cand_s = cand_s2
                        if debug:
                            print(f"[auto] window {ws/feat_cfg.sr:.3f}-{we/feat_cfg.sr:.3f}s backoff: p{p_try:.0f} -> thr={local_thr:.3f}, segs={len(cand_s)}")
                        break
        # shift back to full signal index
        breath_intervals.extend([(ws + s, ws + e) for s, e in cand_s])

    breath_intervals = merge_intervals(breath_intervals, gap=int(vad_cfg.merge_gap_ms * vad_cfg.sr / 1000))
    breath_intervals = drop_short(breath_intervals, min_len=int(vad_cfg.min_breath_ms * vad_cfg.sr / 1000))
    return breath_intervals


# -----------------------------
# Version B: no VAD + with ZFF (approx)
# -----------------------------
@dataclass
class ZFFCfg:
    sr: int = 16000
    # ZFF-ish processing
    hp_cut: float = 40.0      # remove DC/very low drift
    lp_cut: float = 900.0     # focus on excitation-related region
    # frame settings for alpha/beta like features (not strict epoch)
    frame_ms: float = 10.0
    hop_ms: float = 5.0
    # thresholds (tunable)
    max_zff_energy: float = 2.0e-5   # alpha-like: low => breath/unvoiced candidate
    max_zff_slope: float = 6.0e-5    # beta-like: low => breath/unvoiced candidate

    # post spectral refine (reuse BreathFeatCfg but apply globally)
    merge_gap_ms: int = 30
    min_breath_ms: int = 60


def butter_bandpass(sr: int, low: float, high: float, order: int = 4):
    ny = 0.5 * sr
    low_n = max(low / ny, 1e-6)
    high_n = min(high / ny, 0.999999)
    b, a = sps.butter(order, [low_n, high_n], btype="band")
    return b, a


def zff_like_signal(x: np.ndarray, cfg: ZFFCfg) -> np.ndarray:
    # A pragmatic "ZFF-like" excitation emphasis:
    # 1) bandpass to remove drift + keep excitation-related band
    b, a = butter_bandpass(cfg.sr, cfg.hp_cut, cfg.lp_cut, order=4)
    y = sps.filtfilt(b, a, x).astype(np.float32)
    return y


def breath_detect_no_vad_with_zff(
    x: np.ndarray,
    zff_cfg: ZFFCfg,
    feat_cfg: BreathFeatCfg
) -> List[Tuple[int, int]]:
    # Step 1: ZFF-like gating to find low-excitation regions
    z = zff_like_signal(x, zff_cfg)
    frame_len = int(zff_cfg.sr * zff_cfg.frame_ms / 1000)
    hop = int(zff_cfg.sr * zff_cfg.hop_ms / 1000)

    frames = frame_signal(z, frame_len, hop)
    # alpha-like: frame energy
    alpha = np.mean(frames**2, axis=1)
    # beta-like: slope magnitude (mean abs diff)
    beta = np.mean(np.abs(np.diff(frames, axis=1)), axis=1)

    cand1 = (alpha <= zff_cfg.max_zff_energy) & (beta <= zff_cfg.max_zff_slope)

    # Convert cand1 to sample intervals
    cand_intervals = []
    for i, c in enumerate(cand1):
        if c:
            s = i * hop
            e = s + frame_len
            cand_intervals.append((s, e))
    cand_intervals = merge_intervals(cand_intervals, gap=int(zff_cfg.merge_gap_ms * zff_cfg.sr / 1000))

    # Step 2: spectral refine (ER/flatness/var) inside those candidate intervals
    out = []
    frame_len2 = int(feat_cfg.sr * feat_cfg.frame_ms / 1000)
    hop2 = int(feat_cfg.sr * feat_cfg.hop_ms / 1000)

    for s, e in cand_intervals:
        seg = x[s:e]
        if len(seg) < frame_len2:
            continue
        frames2 = frame_signal(seg, frame_len2, hop2)
        cand2, _ = spectral_breath_score(frames2, feat_cfg)
        # cand2 -> sample intervals within seg
        seg_intervals = []
        for i, c in enumerate(cand2):
            if c:
                ss = i * hop2
                ee = ss + frame_len2
                seg_intervals.append((ss, ee))
        seg_intervals = merge_intervals(seg_intervals, gap=int(zff_cfg.merge_gap_ms * zff_cfg.sr / 1000))
        seg_intervals = drop_short(seg_intervals, min_len=int(zff_cfg.min_breath_ms * zff_cfg.sr / 1000))
        out.extend([(s + ss, s + ee) for ss, ee in seg_intervals])

    out = merge_intervals(out, gap=int(zff_cfg.merge_gap_ms * zff_cfg.sr / 1000))
    out = drop_short(out, min_len=int(zff_cfg.min_breath_ms * zff_cfg.sr / 1000))
    return out


# -----------------------------
# Optional: apply attenuation on detected breath segments
# -----------------------------
def apply_fade_attenuation(
    x: np.ndarray,
    intervals: List[Tuple[int, int]],
    sr: int,
    atten_db: float = -10.0,
    fade_ms: float = 10.0
) -> np.ndarray:
    y = x.copy()
    g = 10 ** (atten_db / 20.0)
    fade = int(sr * fade_ms / 1000)
    for s, e in intervals:
        s = max(0, s); e = min(len(y), e)
        if e <= s:
            continue
        # build gain envelope with linear fades
        env = np.ones(e - s, dtype=np.float32) * g
        if fade > 0:
            f = min(fade, (e - s) // 2)
            if f > 0:
                env[:f] = np.linspace(1.0, g, f, dtype=np.float32)
                env[-f:] = np.linspace(g, 1.0, f, dtype=np.float32)
        y[s:e] *= env
    return y


# -----------------------------
# Main demo
# -----------------------------
if __name__ == "__main__":
    import argparse, json

    ap = argparse.ArgumentParser()
    ap.add_argument("wav", type=str, help="input wav")
    ap.add_argument("--mode", type=str, choices=["vad_no_zff", "zff_no_vad"], default="vad_no_zff")
    ap.add_argument("--out_wav", type=str, default="out_atten.wav")
    ap.add_argument("--out_json", type=str, default="breath_segments.json")
    ap.add_argument("--out_json_events", type=str, default=None, help="输出合并后的呼吸事件（可选）")
    ap.add_argument("--atten_db", type=float, default=-10.0)
    ap.add_argument("--debug", action="store_true", help="打印自适应阈值与统计信息")

    # Breath feature tuning (no ZFF)
    ap.add_argument("--min_flatness", type=float, default=0.25, help="谱平坦度下限（更大更严格）")
    ap.add_argument("--min_mid_low_ratio", type=float, default=1.2, help="(1-3.5k)/(0-1k) 能量比下限（更小更宽松）")
    ap.add_argument("--auto", action="store_true", help="根据当前音频（在搜索窗口内）自动放宽阈值")
    ap.add_argument("--auto_ratio_percentile", type=float, default=90.0, help="auto 模式下 ratio 分位数（如 90）")
    ap.add_argument("--auto_ratio_min_percentile", type=float, default=80.0, help="auto 回退时允许降到的最小分位数（默认 80）")
    ap.add_argument("--auto_backoff_step", type=float, default=5.0, help="auto 回退步长（分位数），默认 5")
    ap.add_argument("--auto_rms_percentile", type=float, default=20.0, help="auto 模式下噪声 RMS 分位数（默认 20）")
    ap.add_argument("--auto_rms_factor", type=float, default=1.2, help="auto 模式下 min_rms=噪声RMS*factor（默认 1.2）")
    ap.add_argument("--min_breath_ms", type=int, default=60, help="最小呼吸段时长(ms)")
    ap.add_argument("--merge_gap_ms", type=int, default=40, help="合并间隔(ms)，可减少一个呼吸被切成两段")
    ap.add_argument("--event_merge_gap_ms", type=int, default=200, help="呼吸事件合并间隔(ms)，用于统计呼吸事件数")

    # VAD backend & params (only for mode=vad_no_zff)
    ap.add_argument("--vad_backend", type=str, choices=["silero", "webrtcvad"], default="silero")
    ap.add_argument("--vad_window", type=str, choices=["boundary", "non_speech"], default="non_speech",
                    help="VAD 后在何处搜索呼吸：boundary=语音边界附近；non_speech=全非语音段")
    ap.add_argument("--boundary_win_s", type=float, default=0.5, help="boundary 模式下，语音边界两侧搜索窗口(s)")

    ap.add_argument("--vad_repo", type=str, default=None, help="silero-vad 本地仓库路径（未安装包时使用）")
    ap.add_argument("--vad_threshold", type=float, default=0.5, help="silero-vad 阈值")
    ap.add_argument("--vad_min_speech_ms", type=int, default=250, help="silero-vad 最小语音时长(ms)")
    ap.add_argument("--vad_min_silence_ms", type=int, default=100, help="silero-vad 最小静音时长(ms)")
    ap.add_argument("--vad_speech_pad_ms", type=int, default=30, help="silero-vad 语音段 padding(ms)")
    ap.add_argument("--vad_max_speech_s", type=float, default=float("inf"), help="silero-vad 最大语音段时长(s)")
    ap.add_argument("--vad_onnx", action="store_true", help="silero-vad 使用 onnx 版本")
    args = ap.parse_args()

    x, sr = read_audio_mono(args.wav, target_sr=16000)

    feat_cfg = BreathFeatCfg(
        sr=sr,
        min_flatness=float(args.min_flatness),
        min_mid_low_ratio=float(args.min_mid_low_ratio)
    )

    if args.mode == "vad_no_zff":
        vad_cfg = VADCfg(
            sr=sr,
            aggressiveness=2,
            window_mode=str(args.vad_window),
            boundary_win_s=float(args.boundary_win_s),
            min_breath_ms=int(args.min_breath_ms),
            merge_gap_ms=int(args.merge_gap_ms)
        )

        if args.vad_backend == "silero":
            silero_cfg = SileroVadCfg(
                sr=sr,
                threshold=float(args.vad_threshold),
                min_speech_duration_ms=int(args.vad_min_speech_ms),
                max_speech_duration_s=float(args.vad_max_speech_s),
                min_silence_duration_ms=int(args.vad_min_silence_ms),
                speech_pad_ms=int(args.vad_speech_pad_ms),
                onnx=bool(args.vad_onnx),
                vad_repo=args.vad_repo
            )
            speech_intervals = silero_speech_intervals(x, silero_cfg)
            segs = breath_detect_vad_no_zff(
                x,
                vad_cfg,
                feat_cfg,
                speech_intervals=speech_intervals,
                auto_tune=bool(args.auto),
                auto_ratio_percentile=float(args.auto_ratio_percentile),
                auto_ratio_min_percentile=float(args.auto_ratio_min_percentile),
                auto_backoff_step=float(args.auto_backoff_step),
                auto_rms_percentile=float(args.auto_rms_percentile),
                auto_rms_factor=float(args.auto_rms_factor),
                debug=bool(args.debug)
            )
        else:
            segs = breath_detect_vad_no_zff(
                x,
                vad_cfg,
                feat_cfg,
                auto_tune=bool(args.auto),
                auto_ratio_percentile=float(args.auto_ratio_percentile),
                auto_ratio_min_percentile=float(args.auto_ratio_min_percentile),
                auto_backoff_step=float(args.auto_backoff_step),
                auto_rms_percentile=float(args.auto_rms_percentile),
                auto_rms_factor=float(args.auto_rms_factor),
                debug=bool(args.debug)
            )
    else:
        zff_cfg = ZFFCfg(sr=sr, merge_gap_ms=int(args.merge_gap_ms), min_breath_ms=int(args.min_breath_ms))
        # auto 模式下，根据当前音频自适应 ZFF gating 阈值，避免出现“候选全被过滤”的情况
        if args.auto:
            z = zff_like_signal(x, zff_cfg)
            frame_len = int(zff_cfg.sr * zff_cfg.frame_ms / 1000)
            hop = int(zff_cfg.sr * zff_cfg.hop_ms / 1000)
            frames = frame_signal(z, frame_len, hop)
            alpha = np.mean(frames**2, axis=1)
            beta = np.mean(np.abs(np.diff(frames, axis=1)), axis=1)
            # 取低分位数作为阈值（挑选低激励区域）
            zff_cfg.max_zff_energy = float(np.percentile(alpha, 10))
            zff_cfg.max_zff_slope = float(np.percentile(beta, 10))
            if args.debug:
                print(f"[auto] zff_energy thr(p10)={zff_cfg.max_zff_energy:.3e}, zff_slope thr(p10)={zff_cfg.max_zff_slope:.3e}")
        segs = breath_detect_no_vad_with_zff(x, zff_cfg, feat_cfg)

    y = apply_fade_attenuation(x, segs, sr=sr, atten_db=args.atten_db, fade_ms=12.0)

    sf.write(args.out_wav, y, sr)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(
            [{"start_s": s / sr, "end_s": e / sr, "dur_ms": (e - s) * 1000.0 / sr} for s, e in segs],
            f, ensure_ascii=False, indent=2
        )

    events = merge_intervals(segs, gap=int(args.event_merge_gap_ms * sr / 1000))
    if args.out_json_events:
        with open(args.out_json_events, "w", encoding="utf-8") as f:
            json.dump(
                [{"start_s": s / sr, "end_s": e / sr, "dur_ms": (e - s) * 1000.0 / sr} for s, e in events],
                f, ensure_ascii=False, indent=2
            )

    print(f"[OK] mode={args.mode} segments={len(segs)} events={len(events)}")
    print(f"     wrote: {args.out_wav}")
    print(f"     wrote: {args.out_json}")
    if args.out_json_events:
        print(f"     wrote: {args.out_json_events}")
