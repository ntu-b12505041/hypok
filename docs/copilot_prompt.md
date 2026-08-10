# 給 VS Code Copilot 的執行 Prompt

請直接複製以下內容到 VS Code Copilot Agent：

---

你現在要接手一個研究級 ECG 三分類專案。請先完整閱讀 `README.md`、
`docs/research_protocol.md`、`docs/ecgfounder_finetuning.md`、
`configs/mimic.yaml` 與 `configs/ecgfounder_finetune.yaml`，再開始操作。

研究任務是使用 MIMIC-IV-ECG v1.0 波形與提供的 precomputed ECG–血鉀 CSV，
把 12-lead ECG 分成 HypoK（K<3.5）、NK（3.5≤K<5.5）、HyperK（K≥5.5）。
需要比較兩個模型：

1. 從零訓練的 `se_resnet1d_multitask`。
2. 使用官方 12-lead ECGFounder checkpoint 的
   `ecgfounder_multitask` fine-tuning 對照組。

請依序完成以下工作：

1. 先檢查目前 Git 狀態與現有檔案，不要覆蓋使用者尚未提交的修改。
2. 檢查 Python、CUDA、GPU、磁碟與套件環境，執行：
   - `nvidia-smi`
   - `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
3. 建立 virtual environment，安裝 `pip install -e ".[dev,foundation]"`。
4. 請使用者提供或確認：
   - MIMIC-IV-ECG v1.0 root
   - `data/raw/hyperkalemia_data.csv`
   如果路徑未知，不要猜測，也不要下載或公開受 DUA 約束的資料。
5. 將兩個 config 的 `ecg_root` 改成相同的正確路徑，確認兩者
   的 `cohort_csv` 與 `split_csv` 完全相同。
6. 執行 `python scripts/download_ecgfounder.py`，記錄輸出的 SHA-256，填入
   `configs/ecgfounder_finetune.yaml` 的 `model.checkpoint_sha256`。
7. 執行單元測試與 config validation。任何失敗都先找 root cause，不能略過：
   - `python -m pytest`
   - `hypok-ecg validate-config --config configs/mimic.yaml`
   - `hypok-ecg validate-config --config configs/ecgfounder_finetune.yaml`
8. 如果 cohort 尚未建立，只用 baseline config 建立一次：
   - `hypok-ecg build-cohort --config configs/mimic.yaml --workers 16`
   - `hypok-ecg split --config configs/mimic.yaml`
   檢查 waveform/header error、12 導程排除、每類病人數及 patient leakage。不要為
   ECGFounder 重新產生 split。
9. 在 GPU 上依序訓練與評估：
   - `hypok-ecg train --config configs/mimic.yaml`
   - `hypok-ecg evaluate --config configs/mimic.yaml`
   - `hypok-ecg train --config configs/ecgfounder_finetune.yaml`
   - `hypok-ecg evaluate --config configs/ecgfounder_finetune.yaml`
10. 執行 paired comparison：
    `hypok-ecg compare --baseline-config configs/mimic.yaml --finetune-config configs/ecgfounder_finetune.yaml`
11. 檢查兩個 validation report、training curves、confusion matrix、
    per-class metrics、bootstrap CI、訓練秒數與 comparison report 都存在且可讀。
12. 最後回報：
    - 實際資料版本與樣本數
    - 三個 split 的病人／ECG／class counts
    - 兩模型參數量與訓練時間
    - 每類 recall、specificity、precision、F1
    - overall metrics 與 95% CI
    - 是否每一類 recall 和 specificity 都嚴格大於 0.85
    - 哪個模型較好，以及判斷依據
    - 所有輸出檔案路徑

研究限制：

- 絕對不能捏造訓練或測試結果。
- 絕對不能用 ECG-level random split；必須使用現有 patient-level split。
- 絕對不能用 test set 選 threshold、調參或 early stopping。
- 不要因為 validation 達標就宣稱 test 達標。
- 不要只報 accuracy 或 AUROC，必須逐類報 recall 與 specificity。
- ECGFounder 必須維持 `profile: ecgfounder_official`，包括 500 Hz、10 秒、
  50 Hz notch、0.67–40 Hz bandpass、0.4 秒 median baseline removal 與
  global z-score，不可任意改成 baseline preprocessing。
- 若 GPU、checkpoint、配對 CSV 或對應 ECG 波形不足，停止正式訓練並明確回報
  blocker；仍可完成不需 GPU 的測試，但不得產生假結果。
- 不得將 MIMIC 衍生 cohort、逐筆 predictions 或 checkpoint 推到公開 GitHub。

除非遇到缺少資料路徑、權限或不可逆操作，否則請自主完成、修正錯誤並驗證，
不要每一步都停下來詢問。

---
