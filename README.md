# breath_detect
呼吸声检测

## VAD 前置流程
使用 silero-vad 先得到语音段，再对非语音段进行呼吸检测。

### 快速使用
默认启用 VAD 预处理：
```
python breath_detect_rule.py breath_1.wav --vad-repo E:\Projects\silero-vad
```

禁用 VAD（仅规则法）：
```
python breath_detect_rule.py breath_1.wav --no-vad
```

常用参数：
```
--vad-threshold 0.5
--vad-min-speech-ms 250
--vad-min-silence-ms 100
--vad-speech-pad-ms 30
--vad-max-speech-s 10
--vad-sr 16000
--vad-onnx
```

## 两种方法对比（推荐用这个脚本做 A/B）
脚本：`breath_detect_two_versions.py`

### A) 用 VAD、无 ZFF（推荐）
使用 silero-vad 作为前置，仅在非语音段内做谱规则呼吸检测：
```
python breath_detect_two_versions.py input.wav --mode vad_no_zff --vad_repo E:\Projects\silero-vad --vad_window non_speech --out_wav out_vad.wav --out_json vad_segments.json
```

如果希望参数对不同音频更通用，建议开启自适应（只会“放宽”，不会比你给的阈值更严格）：
```
python breath_detect_two_versions.py input.wav --mode vad_no_zff --vad_repo E:\Projects\silero-vad --vad_window non_speech --auto --out_json vad_segments.json --out_json_events vad_events.json
```

auto 目前做了三件事（目标：减少“换音频就得改参数”）：
- 全局：在搜索窗口内统计 `mid_low_ratio` 分位数，作为阈值上限（并不超过你给的 `--min_mid_low_ratio` 上限）；
- 全局：用非语音窗口的 RMS 分位数估计噪声基线，设置动态 `min_rms`（默认 p20*1.2）；
- 单段兜底：如果某个 non-speech 段在全局阈值下完全检不出，但该段确实有明显声音，会对该段做分位数回退（p90→p85→p80）避免“某一段漏光”。

### B) 不用 VAD、有 ZFF（全段扫）
```
python breath_detect_two_versions.py input.wav --mode zff_no_vad --out_wav out_zff.wav --out_json zff_segments.json
```

### 常见调参
室内播客如果底噪偏 “hiss”（误检偏多）：
- 把 `--min_flatness` 调高一点（更严格），例如 `0.30~0.40`

如果漏检（召回偏低）：
- 先把 `--min_mid_low_ratio` 调低一点（更宽松），例如 `1.0~1.1`，再观察误检变化

auto 模式下如果你发现误检偏多（想更严格一点）：
- 提高 `--auto_ratio_min_percentile`（例如 85/90）
- 或提高 `--auto_rms_factor`（例如 1.3~1.5）

### breath_2.wav（5 段呼吸）针对性建议
默认 `min_mid_low_ratio=1.2` 在 `breath_2.wav` 上容易漏检，推荐先试 auto（更通用）：
```
python breath_detect_two_versions.py .\breath_2.wav --mode vad_no_zff --vad_repo E:\Projects\silero-vad --vad_window non_speech --auto --out_json vad_segments.json --out_json_events vad_events.json --debug
```

如果你仍希望手动锁定阈值（便于可重复对比），可以再试：
```
python breath_detect_two_versions.py .\breath_2.wav --mode vad_no_zff --vad_repo E:\Projects\silero-vad --vad_window non_speech --min_mid_low_ratio 0.6 --min_flatness 0.35 --out_json vad_segments.json
```

分析记录：`docs/analysis-breath_2.md`
