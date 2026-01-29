# breath_2.wav 结果分析记录（VAD vs ZFF）

日期：2026-01-29  
执行者：Codex

## 背景
目标：对 `breath_2.wav` 做呼吸检测，对比两种方法：

- A：`vad_no_zff`（silero-vad 前置，仅在非语音段内用谱特征找呼吸）
- B：`zff_no_vad`（ZFF-like gating + 谱特征精筛）

用户标注/主观判断：`breath_2.wav` 里共有 **5 段**呼吸声。

## 复现命令与现象
### A) VAD + no ZFF（默认阈值）
```
python breath_detect_two_versions.py .\breath_2.wav --mode vad_no_zff --vad_repo E:\Projects\silero-vad --vad_window non_speech --out_json vad_segments.json
```
现象：通常只输出 1 段（例如 10.184–10.374s）。

### B) no VAD + ZFF（默认阈值）
```
python breath_detect_two_versions.py .\breath_2.wav --mode zff_no_vad --out_json zff_segments.json
```
现象：输出 0 段。

## 原因分析
### 1) 为什么 zff_no_vad 为 0 段？
`zff_no_vad` 的第一步 gating 要求：
- `alpha <= max_zff_energy`（默认 `2e-5`）
- `beta <= max_zff_slope`（默认 `6e-5`）

在 `breath_2.wav` 上，ZFF-like slope（beta）的最小值约为 **1.4e-4**，已经大于 `6e-5`，
导致候选帧数量为 0，后续谱特征精筛完全不执行，因此最终为 0 段。

结论：这是“阈值过严导致候选被全部过滤”的典型情况，并不等价于“没有呼吸”。

### 2) 为什么 vad_no_zff 默认只出 1 段？
在非语音段内，谱规则的四个条件里，真正起瓶颈的是：
- `mid_low_ratio >= min_mid_low_ratio`（默认 1.2）

对 `breath_2.wav` 的非语音帧统计显示：
- 非语音帧里，`mid_low_ratio` 的中位数约 0.17，p90 约 0.62，只有极少数帧能达到 1.2；
因此会导致召回偏低，只能检出最“像呼吸”的那一段。

结论：该结果“更偏精确率”，但对你说的 5 段真值来说明显漏检。

## 调参建议（以找回 5 段为目标）
在 `breath_2.wav` 上，将 `min_mid_low_ratio` 放宽到 0.6 左右可以得到约 5 段候选：
```
python breath_detect_two_versions.py .\breath_2.wav --mode vad_no_zff --vad_repo E:\Projects\silero-vad --vad_window non_speech --min_mid_low_ratio 0.6 --min_flatness 0.35 --out_json vad_segments.json
```

注意：
- `min_mid_low_ratio` 越低 → 召回更高，但误检也更可能上升；
- 室内播客如果底噪偏 “hiss”，建议提高 `--min_flatness` 到 0.30~0.40 来抑制误检。

## 自适应与“呼吸事件”输出
为了避免每换一条音频就手动调 `min_mid_low_ratio`，`breath_detect_two_versions.py` 支持 `--auto`：
- 先在所有搜索窗口内统计 `mid_low_ratio` 分布，取分位数（默认 p90）作为 **全局阈值上限**（并且不超过你给的 `--min_mid_low_ratio` 上限）；
- 同时根据非语音窗口的 RMS 统计噪声基线，设置动态 `min_rms`（默认使用 p20 * 1.2）；
- 如果某个 non-speech 段在全局阈值下完全检不出，但该段确实有“明显声音”，会对该段做 **分位数回退**（例如 p90→p85→p80），避免“某一段漏光”。

同时支持 `--out_json_events` 输出合并后的“呼吸事件”（默认 `event_merge_gap_ms=200`），用于减少“一个呼吸被切成多个段”的情况。

### breath_2.wav 在 auto 下的一个可复现结果
```
python breath_detect_two_versions.py .\breath_2.wav --mode vad_no_zff --vad_repo E:\Projects\silero-vad --vad_window non_speech --auto --out_json vad_segments.json --out_json_events vad_events.json --debug
```
现象：`segments=6 events=5`（events 更接近你说的 “5 段呼吸” 的统计口径）。

如果你发现误检偏多：可以提高 `--auto_ratio_min_percentile`（例如 85/90）或提高 `--auto_rms_factor`（例如 1.3~1.5）让阈值更严格。

## 结论（这个结果是否合适？能否正确识别 5 段？）
- 以默认阈值运行：不合适（漏检明显），难以覆盖你说的 5 段。
- 通过放宽 `min_mid_low_ratio`：可以把召回提升到接近 5 段，但仍需结合试听/标注核验误检情况。
- 在 `--auto` 模式下：更倾向于“不同 non-speech 段都能检到候选”，并用 events 口径减少切分，整体更适合作为通用默认方案；是否与标注 5 段完全一致，仍建议用输出时间戳对照核验。
