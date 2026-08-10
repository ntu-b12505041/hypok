# 論文與方法依據

## 最接近本任務的研究

### An et al., Scientific Reports, 2024

**Development of deep learning algorithm for detecting dyskalemia based on
electrocardiogram.** DOI:
[10.1038/s41598-024-71562-5](https://doi.org/10.1038/s41598-024-71562-5)

- 可讀到期刊完整內文、方法與結果，並有作者提供的重實作程式。
- 使用 12 導程 SE-ResNet、10 秒 ECG、resample 至 250 Hz、約 0.67–40 Hz filter。
- HypoK <3.5、HyperK ≥5.5，ECG–K⁺ 配對；validation/test 使用 ±1 小時。
- 它訓練兩個 binary model，不是本專案的直接三分類；本專案因此加上 ordinal 與
  continuous K⁺ 多任務 head。
- 內部 12-lead 報告 AUROC 約 0.929（HyperK）及 0.925（HypoK），但高敏感度
  operating point 的 specificity 約 0.706 與 0.790。這表示「每類 recall 和
  specificity 都 >0.85」比只追求高 AUROC 更嚴格，不能保證達成。
- **重要警示**：期刊頁面顯示 2025-04-08 發布 Editorial Expression of
  Concern。因此本專案只把它當架構與流程參考，不把其結果當無條件可信的效能
  基準；正式論文需揭露此點。

重實作程式：
[bakqui/ecg-dyskalemia](https://github.com/bakqui/ecg-dyskalemia)。
程式標示 non-commercial license，故本專案未複製其程式碼，只重建方法。

### Lin et al., JMIR Medical Informatics, 2020

**A Deep-Learning Algorithm (ECG12Net) for Detecting Hypokalemia and
Hyperkalemia by Electrocardiography.**
[PubMed](https://pubmed.ncbi.nlm.nih.gov/32134388/) /
[PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC7082733/)

- 直接處理 HypoK 與 HyperK，證明 12 導程 ECG 可學到 dyskalemia 訊號。
- 論文頁面與公開全文可取得；原始院內病人資料不能直接取得。
- 對 severe dyskalemia 的表現較高，提醒主要三分類結果之外要報嚴重度分層。

### Galloway et al., JAMA Cardiology, 2019

**Development and Validation of a Deep-Learning Model to Screen for
Hyperkalemia From the Electrocardiogram.**
[Full text](https://jamanetwork.com/journals/jamacardiology/fullarticle/2729582) /
[PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6537816/)

- 針對腎病族群與 HyperK，說明即使 reduced leads 也可能有效。
- 它是 binary HyperK screening，不能直接證明三分類 NK/HypoK/HyperK 的效能。

### Wang et al., Chinese Medical Journal, 2021

**Development and validation of a deep learning model for detection of
hypokalemia from the electrocardiogram in emergency patients.**
[DOI full text](https://mednexus.org/doi/full/10.1097/CM9.0000000000001650)

- 專注急診 HypoK，補足許多研究只看 HyperK 的偏差。
- 可取得文章頁面；原始病人資料未隨本文公開。

### Lin et al., npj Digital Medicine, 2022

**Point-of-care artificial intelligence-enabled ECG for dyskalemia: a
retrospective cohort analysis for accuracy and outcome prediction.**
[Open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC8770475/)

- 同時討論 dyskalemia、point-of-care 與臨床結果。
- 支持報告 class-level sensitivity/specificity，而不是只呈現單一 AUROC。

## 資料來源

- [MIMIC-IV-ECG v1.0](https://physionet.org/content/mimic-iv-ecg/1.0/)：
  約 80 萬張、近 16 萬病人的 12 導程 10 秒 500 Hz ECG，開放下載；官方說明
  `subject_id` 可連結 MIMIC-IV Clinical，也警告 ECG cart clock 可能不同步。
- [MIMIC-IV Clinical v3.1](https://physionet.org/content/mimiciv/3.1/)：
  credentialed access，需要 CITI training 與 DUA；提供 `labevents` 血鉀。

目前主 workflow 使用提供的 precomputed ECG–K⁺ cohort，並明確標示其上游配對
尚未由本專案獨立驗證。本工作環境沒有對應完整波形與 GPU，因此不能在此誠實
產出真實訓練成績。
