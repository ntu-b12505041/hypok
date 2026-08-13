# 研究與驗證計畫

## 研究問題

在具有時間鄰近血清鉀參考值的成人 10 秒 12 導程 ECG 中，端到端深度學習模型
能否將 ECG 分為 HypoK、NK、HyperK，且在鎖定的病人層級測試集上，每一類
recall 與 specificity 都嚴格大於 0.85？

主要用途是 **screening / triage research**，不是以 ECG 取代實驗室檢驗。

## 資料與版本

- MIMIC-IV-ECG v1.0：WFDB 12 導程、10 秒、500 Hz ECG。
- 上游提供的 precomputed cohort：聲明源自 MIMIC-IV Clinical v3.1，並假設
  primary potassium 為 blood chemistry `itemid=50971`。
- 因缺少 `potassium_time`、`itemid`、`labevent_id` 與 `delta_minutes`，本專案
  無法獨立重現或驗證上游 ECG–K⁺ 配對；此限制必須在報告中揭露。
- Secondary sensitivity analysis：whole-blood `50822`、`52452`；啟用前由
  `d_labitems` 自動檢查名稱、fluid 與 category。

MIMIC-IV-ECG 的 `subject_id` 可與 Clinical 對接，且相對於同一病人的時間位移
保持一致。官方同時警告 ECG 機器時鐘可能未同步，因此時間窗分析是必要項目。

## 納入、排除與配對

目前主要分析：

1. `study_id` 唯一，路徑與 study ID 一致。
2. 對應 WFDB header 與 signal file 存在且可讀。
3. CSV 的 ECG time 與 header acquisition time 一致。
4. `n_sig=12` 且導程集合完整為 I、II、III、aVR、aVL、aVF、V1–V6。
5. K⁺ 數值介於 1.5–10.0 mmol/L，提供標籤與固定門檻重新計算結果一致。
6. 上游 ±60 分鐘、最近檢驗與同距離優先 ECG 前檢驗規則視為假設，而非本專案
   已驗證的處理步驟。

建議正式研究再加入成人限制、pacemaker/ICD 排除，以及 ECG 品質排除。這些資訊
必須以可重現 SQL 加入，不能人工挑選。

## 標籤

- HypoK：K⁺ < 3.5 mmol/L
- NK：3.5 ≤ K⁺ < 5.5 mmol/L
- HyperK：K⁺ ≥ 5.5 mmol/L

臨界值在任何資料檢視前固定。主要分析不設灰區，以符合三分類任務；另做
排除 3.4–3.6 與 5.4–5.6 的敏感度分析，量化檢驗誤差及臨界值不穩定性。

## 切分與洩漏防止

- 70% train、15% validation、15% test。
- `subject_id` 是不可分割群組；同一病人的所有 ECG 只能出現在一個 split。
- 依病人是否曾有 HyperK、否則是否曾有 HypoK、否則 NK-only 進行稀有類分層。
- test 在所有模型、threshold、前處理決策鎖定前不可查看。
- 正式論文應另外做 calendar-time split；若能取得另一醫院資料，外部驗證優先。

## 前處理

1. 用 WFDB 讀取 calibrated physical signal（mV）。
2. 固定導程順序 I, II, III, aVR, aVL, aVF, V1–V6。
3. 0.5–40 Hz 四階零相位 band-pass。
4. 500 Hz resample 至 250 Hz。
5. 中央裁切／補零為 10 秒、每導程 2500 samples。
6. 僅在 ±5 mV clip 不合理極端值。
7. 不對每張 ECG 做自身標準差正規化，以免抹除與鉀相關的 T-wave 振幅。

Train-only augmentation：0.01 mV Gaussian noise、最多 80 ms time shift、低機率
lead dropout。預設不做 amplitude scaling。

## 模型與訓練

- Task-specific model：SE-ResNet-34 style 1D CNN，base channels 64，stage blocks
  `[2,2,2,2]`，kernel 7，dropout 0.2。
- Pretrained control：ECGFounder 12-lead Net1D/RegNet-style backbone，使用
  Harvard–Emory ECG Database 預訓練 checkpoint；新建相同三個 downstream
  heads。先凍結 backbone 5 epochs，再全模型 fine-tune；backbone LR 1e-4、
  heads LR 1e-3。
- Heads：3-class logits、2 個 cumulative ordinal logits、continuous K⁺ regression。
- Loss：class-weighted cross entropy + 0.3 ordinal BCE + 0.2 Smooth-L1。
- Class weights：effective number of samples，只由 train 計算。
- Train-only sampling：每個 epoch 保留全部 HypoK/HyperK，並從固定亂序的 NK
  pool 輪替抽取少數類總數的 1.5 倍；約 5 個 epoch 可遍歷全部 NK。每輪實際
  類別與獨立病人數寫入 `logs/sampling_audit.csv`。Validation/Test 不抽樣。
- AdamW，LR 1e-3，weight decay 1e-4，3-epoch warmup + cosine decay。
- Batch 64，最多 60 epochs，validation macro AUROC 選 checkpoint，patience 10。
- Mixed precision 只在 CUDA 啟用。

兩模型使用完全相同的 `ecg_potassium_pairs_split.csv`。ECGFounder 遵循其官方
前處理：12×5000、500 Hz、10 秒及 whole-ECG global z-score；從零訓練模型保留
原始 mV amplitude 並使用 250 Hz。這是預訓練相容性要求，報告中必須明確揭露。

ECG-FM 不作主要 pretrained control，因為公開權重已在 MIMIC-IV-ECG v1.0
預訓練；若 test ECG 也在其自監督預訓練資料中，會形成 transductive pretraining
並使公平比較較難解釋。ECGFounder 的預訓練來源 HEEDB 與本研究 MIMIC cohort
分離，方法上更乾淨。

## 校正與 operating point

Validation-only：

1. 對 classification logits 做 temperature scaling。
2. 比較 classification expected class、ordinal score、K⁺ regression 三個 score。
3. 對 ordered low/high thresholds 進行網格搜尋。
4. 優先選所有類別 recall >0.85 且 specificity >0.85 的方案；若沒有，選
   最小 per-class recall/specificity 最大者。
5. 鎖定 head、temperature 與 thresholds。

Test-only：套用鎖定規則一次，不再修改。

## 指標與統計

主要：

- 每類 recall/sensitivity、specificity、precision、F1、support。
- Accuracy、balanced accuracy、macro/weighted F1。
- Macro one-vs-rest AUROC、macro AUPRC。
- MCC、quadratic weighted kappa。
- 3×3 confusion matrix（count + row-normalized）。

95% CI 以 patient-cluster bootstrap 2,000 次估計，避免同一病人多張 ECG 被視為
獨立樣本。若任何 test 類別樣本太少，報告必須標示 CI 不穩定。

## 預先定義成功與失敗

PASS 僅在獨立 test set 三類的 recall **及** specificity 全部 >0.85。CI 是否也需
高於 0.85 可作更嚴格的次要標準。若未達標，結論是「目標未達」，不能隱藏 NK
或稀有類成績，也不能只報 macro average。

## 必做敏感度與公平性分析

- 配對時間窗：30 / 60 / 120 分鐘。
- ECG 前檢驗 only vs 前後對稱。
- Serum-only vs whole blood included。
- 單一 ECG/病人 vs 所有 ECG。
- Sex、age group、eGFR/CKD、heart rhythm、calendar year。
- 檢驗距離分層、K⁺ 嚴重度、不同 ECG device/cart。

本版程式完成主要分析與病人 bootstrap；次要 subgroup 欄位需在取得受限資料後
加入 cohort，並在不知道 test 結果的前提下預先固定。
