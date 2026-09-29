# 決策紀錄

使用者在 2026-09-29 授權：照任務清單連續執行，需要決定的事由 Claude 自行選擇，事後記錄在這裡。
每筆的格式：編號、日期、步驟、決定、理由、影響範圍。

| # | 日期 | Step | 決定 | 理由 |
|---|---|---|---|---|
| D-001 | 09-29 | 前置 | 程式碼放在獨立的 git repo `/work/b314513067/pi-lfm-code`（**使用者選的**） | 讓 metadata 有真實的 commit；程式碼和 TB 級資料分開 |
| D-002 | 09-29 | 前置 | 路徑：`OFFICIAL_REPO=/work/b314513067/RealPDEBench`、`OUR_REPO=/work/b314513067/pi-lfm-code`、`OLD_DATA_ROOT=/work/b314513067/pi-lfm/data`、`NEW_DATA_ROOT=/work/b314513067/pi-lfm/data_v2.0.1`、`CKPT_ROOT=/work/b314513067/pi-lfm`（官方在 `ckpt/official`，重訓的在 `runs/`）、`RESULTS_ROOT=/work/b314513067/pi-lfm/results`、`HF_ENDPOINT` 不設定（直接連 huggingface.co） | 提出後使用者沒有修正；2.0.0 下載時就是直連 |
| D-003 | 09-29 | 0.1 | 官方 repo 既有的本地 patch（`fluid_hf_dataset.py` sim_id 欄位讀取）**維持現狀**，不還原；每份結果的 metadata 記錄 `official_commit=62f4c80a+patch(sha256)` | 2.0.0 的所有結果都是在這個 patch 下產生，維持現狀新舊才好比較；這個 patch 只影響記憶體和速度，不影響數值。還原它又會讓 real 載入的 RSS 爆掉 |
| D-004 | 09-29 | 0.1 | CNO 不列入範圍（**使用者指定**） | — |
| D-005 | 09-29 | 0.2→5.2 | 實際解析度是 **64×128**，不是規格寫的 128×256。U-Net 採 **4 個解析度層級、3 次下採樣**（64×128→8×16），瓶頸維持規格的 8×16；DiT 預設 **patch 4**（16×32 = 512 token，和規格的 token 數一致），也支援 patch 8（8×16 = 128 token） | 規格的意圖是「瓶頸 8×16、512 token」，而這兩個數字在 64×128 上都還做得到；如果照字面做 4 次下採樣，瓶頸只剩 4×8 |
| D-006 | 09-29 | 0.3 | 不升級官方程式碼 | 上游在 62f4c80 之後沒有新 commit |
| D-007 | 09-29 | 2.1 | numerical 資料不做逐條統計 | 新舊版 numerical 逐位元相同（Step 1.1 的 sha256），統計量必然相同 |
| D-008 | 09-29 | 2.3 | Stage 1 驗證自己重寫（`scripts/stage1_checks.py`），檢查三項：座標、v/vo 正負號慣例、divergence floor（**使用者指示**「沒有就自己建」） | 原本的腳本和報告不存在 |
