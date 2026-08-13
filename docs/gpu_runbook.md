# GPU 正式執行手冊

## 建議資源

- 單張 NVIDIA GPU，至少 16 GB VRAM；24 GB 以上較適合 batch 64。
- 16–32 CPU cores、64 GB RAM。
- MIMIC-IV-ECG 解壓後約數十 GB，另留 checkpoint 與 cache 空間。
- CUDA-compatible PyTorch 2.2+。

若記憶體不足，先把 batch 64 改為 32 或 16；不要改 test set 或臨界值。

## 啟動前檢查

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
hypok-ecg validate-config --config configs/mimic.yaml
```

確認 `torch.cuda.is_available()` 是 `True`，並記錄 driver、CUDA、GPU name。
訓練程式會把環境、總秒數、最佳 epoch 與參數量寫入
`logs/training_summary.json`。

## 執行順序

```bash
hypok-ecg build-cohort --config configs/mimic.yaml --workers 16
hypok-ecg split --config configs/mimic.yaml
hypok-ecg train --config configs/mimic.yaml
hypok-ecg evaluate --config configs/mimic.yaml
```

只有 train split 使用 rotating NK subsampling；validation/test 維持完整原始
分布。訓練期間可查看每輪實際抽樣數：

```bash
tail -n 10 outputs/mimic_se_resnet_multitask/logs/sampling_audit.csv
```

先檢查 cohort summary：

- 三類都有足夠病人；
- `signal_values_validated` 為 `true`；
- `header_errors`、`ecg_time_mismatches`、`incomplete_standard_leads` 與
  `nonfinite_waveforms` 數量合理；
- `matching_independently_verified` 必須如實記錄為 `false`；
- train/validation/test 沒有 patient overlap；
- test HyperK 病人數足以估計窄 CI。

## 實驗治理

1. 第一次正式 run 前保存 config hash 與 split CSV hash。
2. development 只讀 train/validation。
3. 選定單一模型與 threshold 後才執行 `evaluate`。
4. test 未達標就如實報告，不可重複以同一 test 選模型。
5. 若需迭代，建立新的未見 test 或外部 cohort。

## 預估時間

本環境沒有 GPU，故沒有實測訓練時間。實際時間高度取決於配對樣本數、磁碟
吞吐量、GPU 與 workers；報告只會填入正式 run 的 wall-clock 秒數，不使用推測值。
