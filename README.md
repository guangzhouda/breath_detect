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

### B) 不用 VAD、有 ZFF（全段扫）
```
python breath_detect_two_versions.py input.wav --mode zff_no_vad --out_wav out_zff.wav --out_json zff_segments.json
```

### 常见调参
室内播客如果底噪偏 “hiss”（误检偏多）：
- 把 `--min_flatness` 调高一点（更严格），例如 `0.30~0.40`

如果漏检（召回偏低）：
- 先把 `--min_mid_low_ratio` 调低一点（更宽松），例如 `1.0~1.1`，再观察误检变化

### breath_2.wav（5 段呼吸）针对性建议
默认 `min_mid_low_ratio=1.2` 在 `breath_2.wav` 上容易漏检，可先试：
```
python breath_detect_two_versions.py .\breath_2.wav --mode vad_no_zff --vad_repo E:\Projects\silero-vad --vad_window non_speech --min_mid_low_ratio 0.6 --min_flatness 0.35 --out_json vad_segments.json
```

分析记录：`docs/analysis-breath_2.md`
