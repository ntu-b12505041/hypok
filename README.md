# ECG Dyskalemia：HypoK / NK / HyperK

這是一個可重現的研究級專案骨架，以 10 秒、12 導程 ECG 判斷：

- **HypoK**：血清 K⁺ < 3.5 mmol/L
- **NK**：3.5 ≤ K⁺ < 5.5 mmol/L
- **HyperK**：K⁺ ≥ 5.5 mmol/L

主要資料組合是 **MIMIC-IV-ECG v1.0 + MIMIC-IV Clinical v3.1**。前者提供
ECG 波形與 `subject_id`／時間，後者提供同一病人的血鉀檢驗。模型不使用血鉀作為
輸入；血鉀只用來建立 ECG 的真實標籤。

> 目前交付內容沒有真實 MIMIC 訓練結果。本環境沒有 GPU、PyTorch 或受限的
> MIMIC-IV Clinical 資料，因此依要求跳過正式訓練。`outputs/synthetic_demo`
> 只驗證報告與圖表管線，不能視為模型效能。

## 兩個資料集怎麼配合

| 步驟 | MIMIC-IV-ECG | MIMIC-IV Clinical | 產出 |
|---|---|---|---|
| 1 | 讀取 `subject_id`、`study_id`、ECG 時間、WFDB 路徑 | — | ECG 索引 |
| 2 | 以 `subject_id` 與 ECG 時間為中心 | 從 `hosp/labevents` 取血清 K⁺ | ±60 分鐘候選配對 |
| 3 | 每張 ECG 保留時間最近的一筆 K⁺；同距離優先 ECG 前的檢驗 | `itemid=50971` 為主要分析 | 一張 ECG 對一個 K⁺ |
| 4 | ECG 波形是模型輸入 | K⁺ 轉成三類標籤 | 可訓練樣本 |
| 5 | — | `subject_id` 作分組單位 | 病人互斥的 70/15/15 split |

Clinical 資料不是第二種模型輸入。它在主要模型中只負責提供**同時間的參考標準
K⁺**。這能避免模型偷看血液數值，又能把 ECG 正確標成 HypoK、NK 或 HyperK。

## 模型設計

本專案現在有兩個使用相同 cohort、patient split、標籤、loss 與評估方式的模型：

| 角色 | 模型 | 初始化 |
|---|---|---|
| Task-specific baseline | 12-lead SE-ResNet1D multitask | 從零訓練 |
| Pretrained control | ECGFounder 12-lead multitask | HEEDB 預訓練權重 fine-tune |

兩者都有三分類、ordinal、continuous K⁺ regression 三個 head。ECGFounder
對照組先凍結 backbone 5 epochs，再以較小的 backbone learning rate 全模型
fine-tune。ECGFounder 必須使用官方相容的 500 Hz、10 秒、global z-score
前處理，因此兩個模型的 raw ECG 與 split 完全相同，但 model-specific
preprocessing 不同。

## 安裝

建議 Python 3.10–3.12。GPU 機器先依
[PyTorch 官方安裝器](https://pytorch.org/get-started/locally/)安裝符合 CUDA 的
PyTorch，再安裝本專案：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,foundation]"
hypok-ecg validate-config --config configs/mimic.yaml
hypok-ecg validate-config --config configs/ecgfounder_finetune.yaml
```

## 資料準備

1. 下載開放的 [MIMIC-IV-ECG v1.0](https://physionet.org/content/mimic-iv-ecg/1.0/)。
2. 完成 PhysioNet credentialing、CITI 訓練與 DUA，取得
   [MIMIC-IV Clinical v3.1](https://physionet.org/content/mimiciv/3.1/)。
3. 修改 `configs/mimic.yaml` 的 `ecg_root` 與 `clinical_root`。
4. 不要把原始 MIMIC、配對 cohort、逐筆 prediction 或 checkpoint 公開上傳。

## 完整執行

從專案根目錄執行：

```bash
hypok-ecg index-ecg --config configs/mimic.yaml --workers 16
hypok-ecg build-cohort --config configs/mimic.yaml
hypok-ecg split --config configs/mimic.yaml
hypok-ecg train --config configs/mimic.yaml
hypok-ecg evaluate --config configs/mimic.yaml
```

最後一步產生：

- `reports/validation_report.md`
- `metrics/test_metrics.json`
- `metrics/test_confidence_intervals.json`
- `metrics/test_predictions.csv`
- `figures/confusion_matrix.png`
- `figures/training_curves.png`
- `figures/per_class_metrics.png`
- `figures/roc_pr_curves.png`
- `logs/training_summary.json`（含訓練秒數、環境與參數量）

## ECGFounder fine-tuning 對照組

先下載官方 12-lead checkpoint：

```bash
python scripts/download_ecgfounder.py
```

命令會印出 SHA-256。把該值填入
`configs/ecgfounder_finetune.yaml` 的 `model.checkpoint_sha256`，再執行：

```bash
# 不要重新 split；必須沿用 baseline 的 split CSV
hypok-ecg train --config configs/ecgfounder_finetune.yaml
hypok-ecg evaluate --config configs/ecgfounder_finetune.yaml

hypok-ecg compare \
  --baseline-config configs/mimic.yaml \
  --finetune-config configs/ecgfounder_finetune.yaml
```

比較報告會放在 `outputs/model_comparison/`，並檢查兩個 config 指向同一份
split CSV，再記錄 split SHA-256。

## 成功條件

唯一的 PASS 定義是：**鎖定後的 test set 中，每一類 recall > 0.85 且
specificity > 0.85**。Validation 達標不算成功；AUROC 高也不等於這個條件成立。
由於相關大型研究在高敏感度 operating point 的 specificity 未必達 0.85，這是
有挑戰的研究目標，不能事前保證。

若未達標，先做誤差與資料品質分析，再比較：

- ±30、±60、±120 分鐘配對敏感度分析；
- serum-only（主要）與 serum + whole-blood（次要）；
- 只保留時間在 ECG 前的檢驗；
- severe-vs-all 與三分類；
- 預訓練 backbone、focal loss、balanced sampler；
- 年份切分與外院驗證。

不能用 test set 重複選 threshold 或模型；任何修改都必須回到新的 development
cycle，保留新的最終 test 或外部驗證。

## 測試與無 GPU 報告管線

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/make_synthetic_demo.py --output-dir outputs/synthetic_demo
```

第二個命令只會建立明確標示為 synthetic 的示範報告，不會訓練模型。

## 研究文件

- [研究與驗證計畫](docs/research_protocol.md)
- [論文與方法依據](docs/literature_review.md)
- [資料欄位與配對規則](docs/data_dictionary.md)
- [GPU 正式執行手冊](docs/gpu_runbook.md)
- [ECGFounder fine-tuning 說明](docs/ecgfounder_finetuning.md)
- [給 VS Code Copilot 的執行 Prompt](docs/copilot_prompt.md)

本專案僅供研究，不是醫療器材，不能取代抽血檢驗或臨床判斷。
