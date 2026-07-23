# ECGFounder fine-tuning 對照組

## 為什麼選 ECGFounder

ECGFounder 是 12-lead ECG foundation model，以 Harvard–Emory ECG Database
超過一千萬張 ECG 和 150 種 ECG annotation 預訓練。公開 checkpoint 接受
`12 × 5000`，正好對應 500 Hz、10 秒的 MIMIC-IV-ECG。

本研究沒有選 ECG-FM 作主要 control，因為 ECG-FM 的公開 checkpoint 已在
MIMIC-IV-ECG v1.0 預訓練。即使沒有使用血鉀 label，若 test ECG waveform 曾
出現在預訓練，仍屬 transductive pretraining，論文審查時需要特別處理。
ECGFounder 的 HEEDB 預訓練來源與 MIMIC downstream cohort 分離。

## 架構

```text
MIMIC ECG (12 × 5000)
        ↓
ECGFounder pretrained Net1D backbone
        ↓
1024-dimensional embedding
        ├── 3-class head
        ├── ordinal head
        └── continuous K⁺ regression head
```

三個 downstream heads 與 SE-ResNet baseline 相同，因此主要差異是 pretrained
representation 與 backbone architecture。

## 訓練階段

### Epoch 1–5：head warm-up

- Freeze ECGFounder backbone。
- 只更新 classification、ordinal、regression heads。
- 目的：避免隨機初始化的 heads 在第一批 gradient 破壞 pretrained feature。

### Epoch 6 之後：full fine-tuning

- Unfreeze 全部 backbone。
- Backbone LR：`1e-4`。
- Head LR：`1e-3`。
- 使用 AdamW、warmup、cosine decay、early stopping。

## 前處理差異

| 項目 | Scratch SE-ResNet | ECGFounder |
|---|---:|---:|
| Sampling rate | 250 Hz | 500 Hz |
| Samples | 2500 | 5000 |
| Band-pass | 0.5–40 Hz | 0.67–40 Hz |
| Extra filtering | 無 | 50 Hz notch + 0.4 s median baseline |
| Normalization | 保留 mV amplitude | Whole-ECG global z-score |
| Patient split | 相同 | 相同 |

不能把 ECGFounder 改成 250 Hz 後仍稱作正式 checkpoint-compatible fine-tuning；
也不能為兩個模型重新抽不同 split。

## 執行

```bash
python -m pip install -e ".[dev,foundation]"
python scripts/download_ecgfounder.py
```

把下載命令印出的 SHA-256 填入 config：

```yaml
model:
  checkpoint_path: "checkpoints/ecgfounder/12_lead_ECGFounder.pth"
  checkpoint_sha256: "<貼上 SHA-256>"
```

確認 GPU 與 config：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
hypok-ecg validate-config --config configs/ecgfounder_finetune.yaml
```

訓練與評估：

```bash
hypok-ecg train --config configs/ecgfounder_finetune.yaml
hypok-ecg evaluate --config configs/ecgfounder_finetune.yaml
hypok-ecg compare \
  --baseline-config configs/mimic.yaml \
  --finetune-config configs/ecgfounder_finetune.yaml
```

## 公平比較規則

1. 兩模型使用同一份 split CSV。
2. 兩模型的 validation calibration 分開 fitting。
3. 兩模型都只在鎖定後查看 test 一次。
4. 不能因為 test 上其中一個模型輸了就回頭調參後再測同一 test。
5. 主要比較每類 recall、specificity、macro AUPRC 和 patient-bootstrap CI。
6. 訓練時間、參數量與 GPU 也必須一起報告。

Checkpoint 不包含在專案壓縮檔，必須由官方 Hugging Face repository 下載。
