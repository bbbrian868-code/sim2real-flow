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
| D-009 | 09-29 | 5.1 | FM 採方案 **B**（自寫訓練迴圈，import 官方的 dataset、normalizer、metrics） | 方案 A 要改官方的 `load_model.py`，而且官方迴圈沒有 EMA、DDP、bf16、AdamW、warmup |
| D-010 | 09-29 | 5.2 | FM 的 `mask_prob=0.5`、`noise_scale=0.1`（照官方 **U-Net** cylinder config） | 官方各 baseline 的 mask_prob 不同（U-Net 0.5，其他 0.1）；FM 主要和 U-Net 比，所以取 U-Net 的值 |
| D-011 | 09-29 | 6.1 | FM 的 val 子集 = real val 每 9 筆取 1 筆，共 **536 筆**（依 index 順序，涵蓋所有 sim） | val 和 test 共用全部 92 個 sim，而且 75% 的時間窗重疊（Step 1.3），用小子集就足以代表；N=10、K=1 時驗證成本低 |
| D-012 | 09-29 | 5.2 | 大小：U-Net base_ch 40/80/144（channel_mult 1-2-4-4，2 個 res block）= 9.74M/38.80M/125.45M；DiT patch 4，dim×depth = 256×8 / 512×8 / 768×11 = 10.47M/40.21M/121.09M | 同一檔差距在 ±7.5% 內；目標是 10M/40M/120M（附錄 B 第 3 項的預設值） |
| D-013 | 09-29 | 5.2 | FM 訓練超參數：AdamW、lr 1e-4、weight_decay 0、betas (0.9, 0.999)、warmup 1000、global batch 64、clip 1.0、EMA 用 warmup 公式；iteration 預算等 Step 5.5 之後再定 | 規格沒指定 lr 和 wd，採 DiT/ADM 的常用值；wd=0 避免在 EMA 之外再多一個正則化變因 |
| D-014 | 09-29 | 3.x | 所有評估都走 `pilfm/official_eval.py`（逐行複製 eval.py 的迴圈，指標呼叫官方的 `eval_metrics`），不直接執行官方 `eval.py` | 同一條路徑要評估 persistence 和 FM；官方 eval.py 在 log 裡只印 5 位小數，而且同一秒啟動的 run 會寫進同一個目錄。一致性由 Step 3.1 控制組驗證 |
| D-015 | 09-29 | 5.2 | FM 訓練使用 `pilfm/fast_dataset.py`（官方 dataset 的 zero-copy 子類別） | 官方的 `__getitem__` 每筆樣本要解碼約 1.5 GB，batch 64 無法接受；逐位元等價性由測試 (f) 驗證 |
| D-016 | 09-29 | 5.2 | U-Net 的 ResBlock 第二個卷積、bottleneck attention 的輸出投影也做 zero-init（ADM 慣例）；最後一層卷積照規格 zero-init | 這是標準 ADM 結構的一部分，不屬於 5.0 表格裡的「技巧」；它讓初始狀態下每個 block 都等於恆等映射 |
| D-017 | 09-29 | 4.1 | **real 與 finetune 的 seed 0 不重訓**：沿用既有 run 的 50 個中間 checkpoint，在 2.0.1 val 上重新選最佳點；只新訓 seed 1、2 | 2.0.1 裡 real **train** 資料逐位元沒變（`3656.h5` 只在 val/test，Step 1.3），官方流程在同一個 seed 下重訓會得到相同的權重軌跡，唯一的差別是 checkpoint 選擇。finetune 例外：如果 numerical 的最佳點改變，起點就不同，要重訓 |
| D-018 | 09-29 | 4.1 | baseline 重訓的 config 除了資料路徑和 seed，還改了 `num_workers: 6` 和 `exp_name` 後綴 `_s{seed}`，以及 results_path | num_workers 6 是 2.0.0 重訓時避免 cgroup OOM（SIGBUS）用的設定，不影響 real 資料的 batch 組成；exp_name 只影響輸出路徑，用來避免同秒啟動的 run 撞到同一個目錄 |
| D-019 | 09-30 | 5.4 | **DiT 預設 patch 改為 2**（規格是 8，也支援 4）。size 不變：dim×depth = 256×8 / 512×8 / 768×11 → 9.92M / 39.10M / 119.43M | Step 5.4：patch 4 的 DiT-M 在固定 batch 上 loss 卡在 0.50（t=0 時就 0.53）。每個 token 的 y_t 有 p²·60 = 960 維 > hidden 512，patch embedding 無法把 y_t 的逐像素白噪聲傳到輸出，理論下限約 0.5，和觀測吻合。patch 8 時是 3840 維，更差。patch 2 時是 240 維，比各檔的 hidden 都小。**這偏離了 5.2 的規格** |
| D-020 | 09-30 | 5.4 | 過擬合測試加做 warmup 100 的版本（2000 步照規格，另加 10000 步），並加上逐 t 分段的診斷（`fm/tests/diag_overfit.py`） | 原本的排程是 warmup 1000 加 cosine 到 0，2000 步裡有效學習不到一半 |
| D-021 | 09-30 | 4.2 | **finetune 起點**（附錄 B 第 1 項）= 我們重訓的 numerical seed 0，**依 2.0.1 val 重新選出的最佳點**：U-Net `model_9800`、DeepONet `model_1300`、Transolver `model_1100` | 官方 numerical checkpoint 是用有問題的 2.0.0 val 選出來的，而且沒有中間 checkpoint 可以重新選點；我們的 run 有 50 個中間 checkpoint |
| D-022 | 09-30 | 4.1 | 因為 numerical 的最佳點改變了（Step 3.4），**finetune 的 seed 0 也重訓**（D-017 的例外）；real 的 seed 0 仍沿用既有 run，在 2.0.1 val 上重新選點（U-Net 6800、DeepONet 4700、Transolver 5000） | finetune 的起點不同，就不能沿用舊的權重軌跡 |
| D-023 | 09-30 | 5.5 | **FM 訓練預算**（附錄 B 第 2 項）：numerical 和 real 都是 **50k 步** × global batch 64（= 320 萬筆樣本；numerical train 約 176 個 epoch，real 約 353 個 epoch）；val 每 2000 步一次 | 如果照 U-Net baseline 的 10k 步 × batch 64，Step 5.4 顯示光是記住一個 batch，U-Net-M 就要大約 10k 步，顯然不夠。50k 步時 M 檔單張 H200 只要 1.4–2.5 小時，成本很低；checkpoint 由 val 選，不怕過擬合 |
| D-024 | 09-30 | 6.2 | **FM finetune**（附錄 B 第 4 項）：用 numerical 最佳 checkpoint 的 **EMA 權重**初始化，重置 optimizer 和 EMA 的步數計數，**lr × 0.3**，**20k 步**，warmup 同樣是 1000 | lr 的比例沒有官方參考值（baseline 的 finetune 用的是相同 lr），0.3 是常見的中間值；real train 只有 9063 筆，20k × 64 ≈ 141 個 epoch |
| D-025 | 09-30 | 6.1 | 算力（附錄 B 第 6 項）：全部在 Nano4 H200 上跑；M 檔每個 run 用 **1 張 H200**，L 檔用 2 張（DDP） | 本叢集沒有 L40S；M 檔單卡 1.4–2.5 小時就夠 |
| D-026 | 10-02 | 4 / 6.4 | **Update Ratio 的算法**（使用者同意）：官方定義 = N₁/N₂（https://realpdebench.github.io/metrics/data-oriented/），只用於 finetune。採用：**2.0.1 val RMSE**、官方評估間隔（num_update/50；FM 是每 2000 步）；RMSE₀ = real run 的最佳 val RMSE，N₂ = 該點的 iteration；N₁ = finetune **第一次** val RMSE ≤ RMSE₀ 的 iteration；real 和 finetune **按 seed 配對**；從未達到就標 **not reached**。腳本：`scripts/update_ratio.py` | 官網沒有規定用 val 還是 test、評估間隔、達不到時怎麼處理。用官方 checkpoint 存的 val 曲線試過 14 種指標，以及「第一次達到」和「最佳點相除」兩種算法，**都無法同時重現論文的 0.3636 / 0.5758 / 1.0**（照字面算是 0.030 / 0.515 / 0.125），所以和論文的值不可直接比較 |
| D-027 | 10-02 | 6.2 | `unet_M_real_s2`（job 484657）第一次碰到 GPU guard 要換節點重送，但重送時撞到 QOSMaxSubmitJobPerUserLimit 而失敗，run 從未開始。**用完全相同的指令重新送出**（job 485902）。`slurm/gpu_guard.sh` 改成撞到上限時每 60 秒重試，最多 2 小時。另外 submitter 在 15:10 左右隨連線一起中斷，`dit_L_finetune_s0` 的 eval 沒有送出；改用 `setsid nohup` 重啟 | 同一個設定、同一個 seed，重送不影響結果；guard 原本的設計就是要「換節點重跑」，不應該因為送件上限而把 run 弄丟 |
