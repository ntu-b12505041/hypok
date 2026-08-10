# 資料欄位與配對規則

## 目前主要輸入：預先配對 cohort

| 提供欄位 | 專案欄位 | 用途 |
|---|---|---|
| `subject_id` | `subject_id` | 病人層級切分 |
| `study_id` | `study_id` | 唯一 ECG 與 metadata 對照 |
| `ecg_time` | `provided_ecg_time` | 與 WFDB header acquisition time 核對 |
| `path` | `record_path` | 對應 `.hea/.dat` record base path |
| `potassium_value` | `potassium` | 上游已配對的 K⁺ 標籤值 |
| `k_label` | `label` | 必須與專案門檻重新計算結果一致 |

此檔案未提供 `potassium_time`、`itemid`、`labevent_id` 或 `delta_minutes`，因此
專案無法獨立驗證上游 ±60 分鐘最近檢驗規則。`build-cohort` 只驗證本地 ECG
可讀性、時間、12 個標準導程、血鉀範圍和標籤一致性。

## 輸入

### ECG index

| 欄位 | 來源 | 用途 |
|---|---|---|
| `subject_id` | `record_list.csv` | 與 Clinical 連接及病人分組 |
| `study_id` | `record_list.csv` | 唯一 ECG |
| `record_path` | `record_list.csv` | WFDB `.hea/.dat` 路徑 |
| `ecg_time` | WFDB header | 與 K⁺ 配對 |
| `sampling_rate` | WFDB header | resampling |
| `signal_length` | WFDB header | 品質檢查 |
| `n_sig`, `lead_names` | WFDB header | 12 導程檢查 |
| `index_error` | pipeline | 保留不可讀紀錄的稽核原因 |

### Clinical potassium

以下是使用原始 Clinical workflow 重現配對時才需要的欄位；目前 precomputed
workflow 不會讀取這些資料。

| 欄位 | 來源 | 用途 |
|---|---|---|
| `subject_id` | `hosp/labevents` | 連接 ECG |
| `labevent_id` | `hosp/labevents` | deterministic tie-break |
| `itemid` | `hosp/labevents` | serum / whole-blood K⁺ |
| `charttime` | `hosp/labevents` | 時間距離 |
| `valuenum` | `hosp/labevents` | K⁺ 參考值 |
| `valueuom` | `hosp/labevents` | 單位稽核 |
| `flag`, `priority` | `hosp/labevents` | 品質與敏感度分析 |

`d_labitems` 會在 cohort 建立前驗證 item ID 的名稱包含 Potassium，避免版本錯誤
或欄位誤用。

## 配對輸出

| 欄位 | 定義 |
|---|---|
| `potassium` | 最近一筆合格 K⁺ |
| `potassium_time` | 該檢驗的 charttime |
| `delta_minutes` | K⁺ 時間 − ECG 時間；負值表示先抽血 |
| `potassium_itemid` | 實際使用 item ID |
| `label_id` | 0 HypoK、1 NK、2 HyperK |
| `label` | 可讀類別名稱 |
| `split` | train / validation / test |

`study_id` 在主要 cohort 必須唯一；`subject_id` 可重複，但只能出現在一個 split。

## 不得公開

MIMIC-IV Clinical 衍生的 cohort、逐筆 test predictions、checkpoint 都應視為受
DUA 約束的敏感衍生資料。公開程式、空白 config、聚合指標與不含可重識別小群組
的圖表前，仍需依 PhysioNet 規範審查。
