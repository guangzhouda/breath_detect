# 验证记录

日期：2026-01-28  
执行者：Codex

## 验证结论
- 单元/冒烟/功能测试通过
  - 已覆盖 breath_detect_rule.py 与 breath_detect_two_versions.py 的基本可运行性

## 风险说明
- 无阻塞风险。注意运行时会出现 `torch.jit.load` 的弃用警告，不影响当前功能。

---

日期：2026-01-29  
执行者：Codex

## 验证结论
- 单元/冒烟/功能测试通过（`Ran 5 tests`）
- `breath_detect_two_versions.py` 的 `--auto` 增强后，可在 `breath_2.wav` 上得到 `events=5`（与“5 段呼吸”口径更一致，详见 `docs/analysis-breath_2.md`）

## 风险说明
- `--auto` 会对“某些 non-speech 段”做分位数回退以兜底召回；如果你遇到误检偏多，可提高 `--auto_ratio_min_percentile` 或 `--auto_rms_factor` 收紧。
