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
hypok-ecg index-ecg --config configs/mimic.yaml --workers 16
hypok-ecg build-cohort --config configs/mimic.yaml
hypok-ecg split --config configs/mimic.yaml
hypok-ecg train --config configs/mimic.yaml
hypok-ecg evaluate --config configs/mimic.yaml
```

先檢查 cohort summary：

- 三類都有足夠病人；
- `median_abs_time_delta_minutes` 合理；
- `potassium_item_dictionary` 名稱與 fluid 正確；
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
