# 資料欄位與配對規則

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
