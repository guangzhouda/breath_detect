import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import soundfile as sf
from scipy import signal, integrate


def zero_frequency_filtering(x: np.ndarray, sr: int) -> np.ndarray:
    """
    零频率滤波（ZFF）算法提取激励源信息
    参考：Murty & Yegnanarayana (2008)
    """
    # 差分消除直流分量
    x_diff = np.diff(x)
    
    # 通过两次积分器（模拟零频率谐振器）
    # 第一级积分器
    y1 = np.cumsum(x_diff)
    # 移除趋势（线性分量）
    t = np.arange(len(y1)) / sr
    A = np.vstack([t, np.ones(len(t))]).T
    m, c = np.linalg.lstsq(A, y1, rcond=None)[0]
    y1_detrended = y1 - (m * t + c)
    
    # 第二级积分器
    y2 = np.cumsum(y1_detrended)
    # 再次移除趋势
    A = np.vstack([t, np.ones(len(t))]).T
    m, c = np.linalg.lstsq(A, y2, rcond=None)[0]
    zff_signal = y2 - (m * t + c)
    
    return zff_signal


def epoch_detection(zff_signal: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    从ZFF信号中检测时点（epoch）位置
    时点是ZFF信号负斜率过零点
    
    论文中使用峰值检测方法
    """
    # 找到ZFF信号的峰值（负向过零点）
    peaks, properties = signal.find_peaks(-zff_signal, 
                                         height=np.std(zff_signal)*0.5,
                                         distance=int(0.005 * sr))  # 至少5ms间隔
    
    # 将峰值位置作为时点
    return peaks


def compute_alpha_beta(x: np.ndarray, sr: int, epochs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算α（ZFF信号能量）和β（时点处激励强度）
    论文中使用2ms窗口
    """
    zff_signal = zero_frequency_filtering(x, sr)
    
    # 窗口长度（2ms）
    win_len_samples = int(0.002 * sr)
    half_win = win_len_samples // 2
    
    alpha_values = np.zeros(len(epochs))
    beta_values = np.zeros(len(epochs))
    
    for i, epoch_idx in enumerate(epochs):
        # 确保窗口在信号范围内
        start_idx = max(0, epoch_idx - half_win)
        end_idx = min(len(zff_signal), epoch_idx + half_win)
        
        if end_idx <= start_idx:
            alpha_values[i] = 0
            beta_values[i] = 0
            continue
            
        # 提取窗口内的ZFF信号
        win_zff = zff_signal[start_idx:end_idx]
        
        # α：ZFF信号能量（论文公式）
        if len(win_zff) > 0:
            alpha_values[i] = np.mean(win_zff ** 2)
        else:
            alpha_values[i] = 0
        
        # β：ZFF信号在时点处的斜率
        # 使用中心差分计算斜率
        if epoch_idx > 1 and epoch_idx < len(zff_signal) - 1:
            beta_values[i] = (zff_signal[epoch_idx + 1] - zff_signal[epoch_idx - 1]) / 2
        else:
            beta_values[i] = 0
    
    return alpha_values, beta_values


def compute_hngd_features(x_segment: np.ndarray, sr: int) -> Dict[str, float]:
    """
    计算HNGD谱特征：F_D, S_D, E_R, S_V
    参考论文：Yegnanarayana & Dhananjaya (2013)
    """
    # 零时间加窗（20ms窗口）
    window_len = int(0.02 * sr)
    if window_len % 2 == 0:
        window_len += 1
    
    # 汉明窗
    window = signal.windows.hamming(window_len)
    
    # 零时间加窗：在信号开始处应用窗口
    if len(x_segment) < window_len:
        pad_len = window_len - len(x_segment)
        x_padded = np.pad(x_segment, (0, pad_len), mode='constant')
        x_windowed = x_padded * window
    else:
        x_windowed = x_segment[:window_len] * window
    
    # 计算FFT
    n_fft = 2048  # 固定长度，确保一致性
    X = np.fft.rfft(x_windowed, n=n_fft)
    freqs = np.fft.rfftfreq(n=n_fft, d=1/sr)
    
    # 计算群延迟
    phase = np.unwrap(np.angle(X))
    group_delay = -np.diff(phase)
    group_delay = np.append(group_delay, group_delay[-1])  # 保持长度一致
    
    # 计算Hilbert包络
    analytic_signal = signal.hilbert(np.abs(X))
    hilbert_envelope = np.abs(analytic_signal)
    
    # HNGD谱
    hngd_spectrum = group_delay * hilbert_envelope
    
    # 1. F_D: 主导共振频率（HNGD谱最大值对应的频率）
    if len(hngd_spectrum) > 0:
        dominant_idx = np.argmax(np.abs(hngd_spectrum))
        F_D = float(freqs[dominant_idx])
        
        # 2. S_D: 主导共振强度（归一化的HNGD最大值）
        S_D = float(np.abs(hngd_spectrum[dominant_idx]) / (np.sum(np.abs(hngd_spectrum)) + 1e-12))
    else:
        F_D = 0.0
        S_D = 0.0
    
    # 计算功率谱用于E_R和S_V
    power_spectrum = np.abs(X) ** 2
    
    # 3. E_R: 中低频能量比 = E(1000-3500Hz) / E(0-1000Hz)
    low_freq_mask = (freqs >= 0) & (freqs <= 1000)
    mid_freq_mask = (freqs > 1000) & (freqs <= 3500)
    
    E_low = np.sum(power_spectrum[low_freq_mask])
    E_mid = np.sum(power_spectrum[mid_freq_mask])
    
    if E_low > 0:
        E_R = float(E_mid / E_low)
    else:
        E_R = 0.0
    
    # 4. S_V: 频谱方差（归一化）
    if len(power_spectrum) > 0:
        power_normalized = power_spectrum / (np.sum(power_spectrum) + 1e-12)
        mean_power = np.mean(power_normalized)
        S_V = float(np.mean((power_normalized - mean_power) ** 2))
    else:
        S_V = 0.0
    
    return {
        "F_D": F_D,
        "S_D": S_D,
        "E_R": E_R,
        "S_V": S_V
    }


@dataclass
class PaperAlgorithmConfig:
    """
    论文算法的配置参数（全部来自论文表1）
    注意：这些都是固定的经验阈值，不需要训练
    """
    # === 论文表1中的阈值 ===
    T_alpha: float = 0.015      # α阈值
    T_beta: float = 0.035       # β阈值
    
    # F_D阈值范围 (Hz)
    T_F_min: float = 0.0
    T_F_max: float = 2800.0
    
    # S_D阈值范围
    T_S_min: float = 0.0001
    T_S_max: float = 0.01
    
    # E_R和S_V阈值
    T_E: float = 26e-5         # E_R阈值（论文：26×10^-5）
    T_V: float = 1.3e-4        # S_V阈值（论文：1.3×10^-4）
    
    # === 后处理参数（论文第6步和第7步）===
    min_segment_gap_ms: float = 20.0    # 合并间隔（论文：20ms）
    min_segment_duration_ms: float = 30.0  # 最小段长（论文：30ms）
    
    # === 信号处理参数 ===
    sr: int = 16000                     # 采样率（论文使用）
    analysis_window_ms: float = 20.0    # 分析窗口
    zff_window_ms: float = 2.0          # ZFF分析窗口（2ms）
    
    debug: bool = True


def paper_breath_detection_algorithm(x: np.ndarray, cfg: PaperAlgorithmConfig) -> Dict[str, Any]:
    """
    实现论文中的7步呼吸音检测算法（基于规则，无训练）
    
    算法步骤：
    1. 对于给定的语音信号（16kHz采样），在时点位置计算α和β值
    2. 选择满足 α ≤ T_α 且 β ≤ T_β 的时点
    3. 为步骤2保留的时点计算F_D和S_D
    4. 选择满足 T_F_min ≤ F_D ≤ T_F_max 且 T_S_min ≤ S_D ≤ T_S_max 的时点
    5. 为步骤4保留的段计算E_R和S_V（每1ms，使用2ms窗口）
    6. 选择满足 E_R > T_E 且 S_V < T_V 的段
    7. 合并间隔小于20ms的段，去除短于30ms的段
    """
    # 计算总时长（所有返回路径都会用到）
    total_duration = len(x) / cfg.sr
    
    if cfg.debug:
        print("=== 开始论文呼吸音检测算法 ===")
        print(f"信号长度: {len(x)} 样本, 时长: {total_duration:.3f}秒")
    
    # 步骤1: 计算ZFF信号和时点
    zff_signal = zero_frequency_filtering(x, cfg.sr)
    epochs = epoch_detection(zff_signal, cfg.sr)
    
    if len(epochs) == 0:
        if cfg.debug:
            print("警告: 未检测到时点")
        return {
            "breath_segments": [],
            "num_breath_segments": 0,
            "coverage": 0.0,
            "total_duration_s": total_duration,
            "breath_duration_s": 0.0,
            "algorithm_steps": {
                "total_epochs": 0,
                "step2_selected": 0,
                "step4_selected": 0,
                "step6_selected": 0,
                "final_segments": 0
            },
            "parameters": asdict(cfg)
        }
    
    if cfg.debug:
        print(f"步骤1: 检测到 {len(epochs)} 个时点")
    
    # 计算α和β值
    alpha_values, beta_values = compute_alpha_beta(x, cfg.sr, epochs)
    
    # 步骤2: 基于α和β阈值选择时点
    step2_mask = (alpha_values <= cfg.T_alpha) & (beta_values <= cfg.T_beta)
    step2_epochs = epochs[step2_mask]
    
    if cfg.debug:
        print(f"步骤2: α≤{cfg.T_alpha}且β≤{cfg.T_beta}的时点: {len(step2_epochs)}个")
    
    if len(step2_epochs) == 0:
        return {
            "breath_segments": [],
            "num_breath_segments": 0,
            "coverage": 0.0,
            "total_duration_s": total_duration,
            "breath_duration_s": 0.0,
            "algorithm_steps": {
                "total_epochs": len(epochs),
                "step2_selected": 0,
                "step4_selected": 0,
                "step6_selected": 0,
                "final_segments": 0
            },
            "parameters": asdict(cfg)
        }
    
    # 步骤3和4: 基于F_D和S_D阈值选择
    step4_segments = []
    
    # 将时点转换为段（相邻时点之间的区域）
    for i in range(len(step2_epochs) - 1):
        start_idx = step2_epochs[i]
        end_idx = step2_epochs[i + 1]
        
        if end_idx - start_idx < int(0.005 * cfg.sr):  # 至少5ms
            continue
        
        # 提取段并计算HNGD特征
        segment = x[start_idx:end_idx]
        features = compute_hngd_features(segment, cfg.sr)
        
        F_D = features["F_D"]
        S_D = features["S_D"]
        
        # 检查是否满足F_D和S_D阈值
        if (cfg.T_F_min <= F_D <= cfg.T_F_max) and (cfg.T_S_min <= S_D <= cfg.T_S_max):
            step4_segments.append({
                "start": start_idx,
                "end": end_idx,
                "F_D": F_D,
                "S_D": S_D
            })
    
    if cfg.debug:
        print(f"步骤4: F_D∈[{cfg.T_F_min},{cfg.T_F_max}]且S_D∈[{cfg.T_S_min},{cfg.T_S_max}]的段: {len(step4_segments)}个")
    
    if len(step4_segments) == 0:
        return {
            "breath_segments": [],
            "num_breath_segments": 0,
            "coverage": 0.0,
            "total_duration_s": total_duration,
            "breath_duration_s": 0.0,
            "algorithm_steps": {
                "total_epochs": len(epochs),
                "step2_selected": len(step2_epochs),
                "step4_selected": 0,
                "step6_selected": 0,
                "final_segments": 0
            },
            "parameters": asdict(cfg)
        }
    
    # 步骤5和6: 基于E_R和S_V阈值选择
    step6_segments = []
    
    for seg_info in step4_segments:
        start_idx = seg_info["start"]
        end_idx = seg_info["end"]
        
        segment = x[start_idx:end_idx]
        features = compute_hngd_features(segment, cfg.sr)
        
        E_R = features["E_R"]
        S_V = features["S_V"]
        
        # 论文条件: E_R > T_E 且 S_V < T_V
        if E_R > cfg.T_E and S_V < cfg.T_V:
            # 转换为时间（秒）
            start_time = start_idx / cfg.sr
            end_time = end_idx / cfg.sr
            step6_segments.append((start_time, end_time))
    
    if cfg.debug:
        print(f"步骤6: E_R>{cfg.T_E}且S_V<{cfg.T_V}的段: {len(step6_segments)}个")
    
    # 步骤7: 后处理
    # 合并间隔小于20ms的段
    merged_segments = []
    if step6_segments:
        step6_segments.sort(key=lambda x: x[0])
        
        current_start, current_end = step6_segments[0]
        
        for seg_start, seg_end in step6_segments[1:]:
            if seg_start - current_end <= cfg.min_segment_gap_ms / 1000.0:
                # 合并
                current_end = max(current_end, seg_end)
            else:
                # 保存当前段
                if current_end - current_start >= cfg.min_segment_duration_ms / 1000.0:
                    merged_segments.append((current_start, current_end))
                current_start, current_end = seg_start, seg_end
        
        # 保存最后一个段
        if current_end - current_start >= cfg.min_segment_duration_ms / 1000.0:
            merged_segments.append((current_start, current_end))
    
    # 计算统计信息
    breath_duration = sum(end - start for start, end in merged_segments)
    coverage = breath_duration / total_duration if total_duration > 0 else 0.0
    
    if cfg.debug:
        print(f"步骤7: 最终呼吸段: {len(merged_segments)}个")
        print(f"总时长: {total_duration:.3f}s, 呼吸时长: {breath_duration:.3f}s")
        print(f"覆盖率: {coverage:.2%}")
        print("=== 算法完成 ===")
    
    return {
        "breath_segments": merged_segments,
        "num_breath_segments": len(merged_segments),
        "coverage": coverage,
        "total_duration_s": total_duration,
        "breath_duration_s": breath_duration,
        "algorithm_steps": {
            "total_epochs": len(epochs),
            "step2_selected": len(step2_epochs),
            "step4_selected": len(step4_segments),
            "step6_selected": len(step6_segments),
            "final_segments": len(merged_segments)
        },
        "parameters": asdict(cfg)
    }


def evaluate_with_ground_truth(detected_segments: List[Tuple[float, float]],
                              ground_truth_segments: List[Tuple[float, float]],
                              total_duration: float,
                              tolerance_ms: float = 50.0) -> Dict[str, float]:
    """
    与真实标注比较评估性能
    
    注意：论文中报告的性能指标：
    - 召回率: 97.29%
    - 精确率: 78.17%
    - F值: 86.69%
    """
    tolerance_s = tolerance_ms / 1000.0
    
    # 初始化统计
    TP = 0  # 真正例（检测到且匹配）
    FP = 0  # 假正例（检测到但不匹配）
    FN = 0  # 假反例（未检测到的真实段）
    
    # 标记已匹配的真实段
    matched_truth = [False] * len(ground_truth_segments)
    
    # 对于每个检测到的段，检查是否与真实段匹配
    for det_start, det_end in detected_segments:
        matched = False
        for i, (truth_start, truth_end) in enumerate(ground_truth_segments):
            if matched_truth[i]:
                continue
            
            # 计算重叠
            overlap_start = max(det_start, truth_start)
            overlap_end = min(det_end, truth_end)
            
            if overlap_end > overlap_start:
                # 有重叠，检查是否在容差范围内
                center_det = (det_start + det_end) / 2
                center_truth = (truth_start + truth_end) / 2
                
                if abs(center_det - center_truth) <= tolerance_s:
                    TP += 1
                    matched_truth[i] = True
                    matched = True
                    break
        
        if not matched:
            FP += 1
    
    # 未匹配的真实段是假反例
    FN = sum(1 for matched in matched_truth if not matched)
    
    # 计算指标
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f_measure = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
    
    return {
        "true_positive": TP,
        "false_positive": FP,
        "false_negative": FN,
        "recall": recall,
        "precision": precision,
        "f_measure": f_measure,
        "expected_recall": 0.9729,  # 论文结果
        "expected_precision": 0.7817,  # 论文结果
        "expected_f_measure": 0.8669  # 论文结果
    }


def main():
    parser = argparse.ArgumentParser(
        description="基于论文的呼吸音检测算法（无训练，基于规则）"
    )
    parser.add_argument("audio_file", help="输入音频文件路径")
    parser.add_argument("--sr", type=int, default=16000, 
                       help="采样率（默认：16000，论文使用）")
    parser.add_argument("--ground-truth", help="真实标注文件（JSON格式）")
    parser.add_argument("--output", help="输出结果文件路径")
    parser.add_argument("--debug", action="store_true", 
                       help="启用详细调试输出")
    parser.add_argument("--compare-paper", action="store_true",
                       help="与论文结果比较")
    
    args = parser.parse_args()
    
    # 加载音频文件
    try:
        x, file_sr = sf.read(args.audio_file)
        if x.ndim > 1:
            x = np.mean(x, axis=1)  # 转为单声道
        
        # 如果需要，重采样到目标采样率
        if file_sr != args.sr:
            from scipy import signal
            x = signal.resample(x, int(len(x) * args.sr / file_sr))
            if args.debug:
                print(f"重采样: {file_sr}Hz -> {args.sr}Hz")
    except Exception as e:
        print(f"加载音频文件失败: {e}")
        sys.exit(1)
    
    # 配置算法参数（全部使用论文中的固定值）
    cfg = PaperAlgorithmConfig(
        T_alpha=0.015,
        T_beta=0.035,
        T_F_min=0.0,
        T_F_max=2800.0,
        T_S_min=0.0001,
        T_S_max=0.01,
        T_E=26e-5,
        T_V=1.3e-4,
        min_segment_gap_ms=20.0,
        min_segment_duration_ms=30.0,
        sr=args.sr,
        debug=args.debug
    )
    
    # 运行论文算法
    result = paper_breath_detection_algorithm(x, cfg)
    
    # 如果有真实标注，进行评估
    if args.ground_truth:
        try:
            with open(args.ground_truth, 'r') as f:
                ground_truth = json.load(f)
            
            truth_segments = ground_truth.get("breath_segments", [])
            if truth_segments:
                evaluation = evaluate_with_ground_truth(
                    result["breath_segments"],
                    truth_segments,
                    result["total_duration_s"]
                )
                result["evaluation"] = evaluation
                
                if args.debug:
                    print("\n=== 性能评估 ===")
                    print(f"检测到段数: {result['num_breath_segments']}")
                    print(f"真实段数: {len(truth_segments)}")
                    print(f"真正例(TP): {evaluation['true_positive']}")
                    print(f"假正例(FP): {evaluation['false_positive']}")
                    print(f"假反例(FN): {evaluation['false_negative']}")
                    print(f"召回率: {evaluation['recall']:.3%}")
                    print(f"精确率: {evaluation['precision']:.3%}")
                    print(f"F值: {evaluation['f_measure']:.3%}")
                    
                    if args.compare_paper:
                        print(f"\n论文结果:")
                        print(f"召回率: {evaluation['expected_recall']:.3%}")
                        print(f"精确率: {evaluation['expected_precision']:.3%}")
                        print(f"F值: {evaluation['expected_f_measure']:.3%}")
        except Exception as e:
            print(f"加载真实标注失败: {e}")
    
    # 输出结果
    if args.output:
        try:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"结果已保存到: {args.output}")
        except Exception as e:
            print(f"保存结果失败: {e}")
    else:
        # 控制台输出简化结果
        print("\n=== 检测结果 ===")
        print(f"音频: {args.audio_file}")
        print(f"时长: {result['total_duration_s']:.3f}秒")
        print(f"检测到呼吸段: {result['num_breath_segments']}个")
        print(f"呼吸时长: {result['breath_duration_s']:.3f}秒")
        print(f"覆盖率: {result['coverage']:.2%}")
        
        if result['breath_segments']:
            print("\n呼吸段列表:")
            for i, (start, end) in enumerate(result['breath_segments'][:10]):  # 只显示前10个
                print(f"  {i+1:3d}. [{start:7.3f}s - {end:7.3f}s], 时长: {end-start:.3f}s")
            if len(result['breath_segments']) > 10:
                print(f"  ... 还有 {len(result['breath_segments'])-10} 个段")


if __name__ == "__main__":
    main()