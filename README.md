# ECG Dyskalemia：HypoK / NK / HyperK

這是一個可重現的研究級專案骨架，以 10 秒、12 導程 ECG 判斷：

- **HypoK**：血清 K⁺ < 3.5 mmol/L
- **NK**：3.5 ≤ K⁺ < 5.5 mmol/L
- **HyperK**：K⁺ ≥ 5.5 mmol/L

主要 workflow 使用一份由 MIMIC-IV Clinical v3.1 預先產生的 ECG–血鉀配對 CSV，
以及 **MIMIC-IV-ECG v1.0** 的實際 WFDB 波形。配對 CSV 只提供標籤與路徑，不能
取代 `.hea/.dat` ECG 檔；模型輸入仍然只有 ECG，血鉀只作為真實標籤。

> 目前交付內容沒有真實 MIMIC-IV 訓練結果。本環境沒有對應的完整 ECG 波形與
> GPU，因此依要求跳過正式訓練。`outputs/synthetic_demo`
> 只驗證報告與圖表管線，不能視為模型效能。

## 配對 CSV 與 ECG 波形怎麼配合

| 步驟 | MIMIC-IV-ECG | Precomputed CSV | 產出 |
|---|---|---|---|
| 1 | 依 `path` 找 `.hea/.dat` | 提供 `subject_id`、`study_id`、K⁺、標籤 | 候選 ECG–K⁺ pair |
| 2 | 讀取 header 的時間、取樣率、導程 | 提供的 ECG time | 時間一致性檢查 |
| 3 | 確認 12 個標準導程與 signal file | K⁺ 重新套用固定門檻 | 合格訓練 cohort |
| 4 | ECG 波形作為模型輸入 | K⁺ 只作真實標籤 | 三分類訓練樣本 |
| 5 | — | 以 `subject_id` 分組 | 病人互斥的 70/15/15 split |

上游配對被假設符合 serum `itemid=50971`、±60 分鐘最近檢驗及同距離優先 ECG
前檢驗；因 CSV 缺少 lab time 與 itemid，本專案不把這些假設描述為已驗證事實。

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

訓練集使用相同的 **rotating NK subsampling**：每個 epoch 保留全部 HypoK 與
HyperK，並抽取約為兩個少數類總數 1.5 倍的 NK。NK 子集合會逐 epoch 輪替，
不永久刪除任何 ECG。Validation 與 Test 完整保留原始盛行率，不做抽樣。

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

1. 將提供的 `hyperkalemia_data.csv` 放到 `data/raw/hyperkalemia_data.csv`。
2. 準備 [MIMIC-IV-ECG v1.0](https://physionet.org/content/mimic-iv-ecg/1.0/)
   對應的 WFDB `.hea` 與 `.dat` 檔；CSV 的 `path` 是相對於 ECG root 的 record path。
3. 修改 `configs/mimic.yaml` 與 `configs/ecgfounder_finetune.yaml` 的 `ecg_root`。
4. `build-cohort` 只讀 CSV 中列出的 ECG header，檢查波形檔存在、ECG 時間一致，
   並只保留完整的 I、II、III、aVR、aVL、aVF、V1–V6。
5. 上游 ECG–K⁺ 配對視為未獨立驗證的研究假設；正式報告會明確揭露。
6. 不要把原始 MIMIC、配對 cohort、逐筆 prediction 或 checkpoint 公開上傳。

若 Server 已經有完整 MIMIC-IV-ECG，不需要再次下載，只要把 `ecg_root` 指向包含
`files/p*/...` 的資料夾。若只有部分 ECG，則必須依 CSV 的 `path` 取得每筆對應的
`.hea` 與 `.dat`，否則該筆會記錄在 `*.excluded.csv` 並排除。

### 依 CSV 選擇性下載 ECG

先確認目的磁碟空間，再用 dry-run 檢查筆數與第一組網址：

```bash
df -h /mnt/sdb1
python scripts/download_selected_mimic4_ecg.py \
  --csv data/raw/hyperkalemia_data.csv \
  --output-root /mnt/sdb1/bdm0162/hypok-data/mimic-iv-ecg/1.0 \
  --dry-run
```

正式下載建議放在 `tmux` 中：

```bash
tmux new -s mimic4-download
python scripts/download_selected_mimic4_ecg.py \
  --csv data/raw/hyperkalemia_data.csv \
  --output-root /mnt/sdb1/bdm0162/hypok-data/mimic-iv-ecg/1.0 \
  --workers 8 \
  2>&1 | tee run_logs/mimic4_selective_download.log
```

下載器使用 `.part` 暫存檔與 HTTP Range 續傳；中斷後執行相同指令即可跳過已完成
檔案並續傳未完成檔案。完成後，兩個模型 config 的 `ecg_root` 都設定為上述
`--output-root`。

## 完整執行

從專案根目錄執行：

```bash
hypok-ecg build-cohort --config configs/mimic.yaml --workers 16
hypok-ecg split --config configs/mimic.yaml
hypok-ecg train --config configs/mimic.yaml
hypok-ecg evaluate --config configs/mimic.yaml
```

先用少量資料驗證路徑與格式：

```bash
hypok-ecg build-cohort --config configs/mimic.yaml --workers 8 --limit 100
```

確認 summary 後，移除 `--limit` 重建正式 cohort 與 split；不得拿 limit 版本正式訓練。

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
- `logs/sampling_audit.csv`（每個 epoch 實際 ECG／病人與各類抽樣數）

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
