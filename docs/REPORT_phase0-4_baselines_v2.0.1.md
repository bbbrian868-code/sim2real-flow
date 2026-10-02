# RealPDEBench cylinder：2.0.1 資料集上的 baseline 復現（Phase 0–4）

- 期間：2026-09-29 ～ 10-01；機器：NCHC Nano4（H200，Slurm，計畫 MST114566）
- 範圍：U-Net / DeepONet / Transolver × Simulated（numerical）/ Real-world（real）/ Sim-pretrain + Real-finetune，主評估 = real test、N_ar = 1
- 各步驟的詳細報告在 `docs/reports/step_*.md`；所有決策（含偏離任務清單的地方）在 `docs/DECISIONS.md`（D-001 ～ D-025）

---

## 0. 摘要

1. **2.0.1 只修正了一條軌跡。** cylinder 範圍內唯一變更的檔案是 `hf_dataset/real/data-00018-of-00073.arrow`，其中只有 `3656.h5` 改變：u、v 做了共同的仿射換算（新 = 0.06838 × 舊 + b），vo 乘以 400，座標從像素改成公尺。numerical 資料、所有 split 和 params JSON、`mean_std.pt` 都逐位元相同。
2. **這條軌跡只佔 test 的 1.78%，卻主導了 2.0.0 的 RMSE、MAE、R²。** 修正後，real/finetune 模型的 RMSE 下降 62–83%，Rel L2 只下降 1–13%。論文 Table 1 的 RMSE 欄因此需要重算，Rel L2 欄大致仍然成立。
3. **排名改變。** 在 2.0.0 上，persistence（直接複製最後一幀）的 RMSE、fRMSE、FE 比所有學習型模型都好；在 2.0.1 上，所有 real/finetune 模型都贏過它，U-Net 領先最多（RMSE 0.0121，DeepONet 0.0234–0.0248、Transolver 0.0238–0.0268，persistence 0.0364）。
4. **「DeepONet / Transolver 很早就達到最佳」的現象在 2.0.1 上消失了。** 例如 Transolver real 的最佳點從 600 變成 5000（最後一步），這是 val 裡那條錯誤軌跡造成的假象。
5. **評估路徑經過驗證。** 自寫的評估 harness 和官方 `eval.py` 的結果逐位元相同（RMSE 一致，13 個指標的最大相對差異 1.6e-7）。
6. **sim 預訓練對 U-Net 沒有幫助**（finetune 0.01209 ± 0.00002 vs real 0.01209 ± 0.00004），**對 Transolver 有幫助**（0.0238 ± 0.0001 vs 0.0268 ± 0.0016，−11%，而且 seed 之間的變異小很多）。

---

## 1. 路徑與版本總覽

| 項目 | 路徑或值 |
|---|---|
| 官方 code | `/work/b314513067/RealPDEBench` @ `62f4c80a`，外加本地 patch（`fluid_hf_dataset.py` 的 sim_id 欄位讀取，只影響記憶體，見 D-003；diff 在 `pi-lfm-code/patches/`） |
| 我們的 code | `/work/b314513067/pi-lfm-code`（git；本報告對應的 commit 見 `git log`） |
| 2.0.0 資料（唯讀） | `/work/b314513067/pi-lfm/data` |
| 2.0.1 資料 | `/work/b314513067/pi-lfm/data_v2.0.1`（HF revision `0ae4426085d2`；181 個未變更的檔案是指向 2.0.0 的 **symlink**，**2.0.0 目錄不能刪或搬**；清單在 `fetch_manifest.json`） |
| 官方 checkpoint | `/work/b314513067/pi-lfm/ckpt/official/cylinder/{unet,deeponet,transolver}/{numerical,real,finetune}.pth` |
| 2.0.0 時期重訓的 run（seed 0） | `/work/b314513067/pi-lfm/runs/{model}/{model}_cylinder_{numerical\|real}_{False\|True}/<timestamp>/` |
| 2.0.1 重訓的 run | `/work/b314513067/pi-lfm/results/v2.0.1/baselines/runs/{model}/{model}_cylinder_s{seed}_real_{False\|True}/<timestamp>/` |
| 結果根目錄 | `/work/b314513067/pi-lfm/results`（以下簡寫為 `$R`） |
| Slurm 輸出 | `$R/slurm/<jobname>-<jobid>.{out,err}`；2.0.0 時期的在 `/work/b314513067/pi-lfm/runs/slurm/` |

**目錄慣例**
- 每個官方訓練 run 的目錄包含：`model_XXXX.pth`（50 個中間 checkpoint，每 num_update/50 步一個）、`training.log`（文字 log，含每次 val 的 14 個指標）、`events.out.tfevents.*`（TensorBoard）。
- 每個 checkpoint 都存有：`model_state_dict`、`train_losses`（每步一個值）、`val_losses`（dict，14 種指標 × 到目前為止的驗證點）、`best_iteration`、`best_val_loss`。
- 每個評估 JSON 都含有：`metrics`（13 個官方指標加上 normalized_mse）、`job`（checkpoint 路徑）、`meta`（兩個 repo 的 commit 和 dirty 狀態、`dataset_version`、config 全文和 sha256、seed、host、slurm job id）。

---

## 2. 評估協議（Step 0.2）

詳見 `docs/reports/step_0.2_protocol.md`（每一項都附官方程式碼的檔名和行號）。

| 項目 | 值 |
|---|---|
| T_in / T_out | 20 / 20；loader 輸出 `(T, H, W, C) = (20, 64, 128, 3)`（real 原生 64×128；numerical 原生 128×256，降採樣 2 倍） |
| 通道 | (u, v, p)，**target 包含 p**；real 的 p 一律填 0，評估時自動只算 u、v（c = 2） |
| sim 加噪 | 只對 numerical，**input 和 target 都加**，乘法型 gaussian，`noise_scale = 0.1` |
| 遮蔽 | 只對 numerical，p 整個設為 0（input 和 target 同時），`mask_prob`：U-Net 0.5，DeepONet/Transolver 0.1 |
| 正規化 | 每個通道的 Gaussian 統計量，**永遠取自 numerical train**（`cylinder/mean_std.pt`；檔案不存在時會靜默重算） |
| 最佳化 | Adam；cosine；U-Net 10k 步、batch 12、lr 1e-4；DeepONet 5k、32、1e-4；Transolver 5k、16、7e-4 |
| checkpoint 選擇 | 每 num_update/50 步在 **real val**（完整）上算**反正規化後的 RMSE**，取最小值 |
| finetune | 只載入權重；optimizer 和 scheduler 重新建立；lr 和步數與從頭訓練相同 |
| 評估 | real test、N_ar = 1、test_mode = all、metric batch = 整個測試集 |

**評估實作**：`pi-lfm-code/pilfm/official_eval.py`（逐行複製 `eval.py:290-352` 和 `train.py:345-373`，指標呼叫官方的 `eval_metrics`），批次執行用 `scripts/eval_ckpts.py`。

**控制組（Step 3.1）**：官方 U-Net numerical 在 2.0.0 上，harness 和**未修改的官方 `eval.py`**（`scripts/official_eval_capture.py` 在執行時攔截指標，不改原始碼）都在 GPU 上跑 → RMSE 逐位元相同，最大相對差異 1.56e-7（門檻是 1e-6）。
- 結果：`$R/v2.0.0/control/official_capture_unet_numerical_gpu.json` vs `$R/v2.0.0/reeval/unet_numerical_official.json`
- 官方 eval 的 log：`$R/v2.0.0/control/official_eval_runs/unet/unet_cylinder_eval/*/eval.log`
- 第一次 capture 跑在 CPU 上（節點問題，見 §7），差異是 2.8e-4，保留在 `official_capture_unet_numerical.json`，僅供參考

---

## 3. 資料變更（Phase 1–2）

| 步驟 | 結論 | 產出 |
|---|---|---|
| 1.1 遠端比對 | 2.0.0 → 2.0.1 的 HF commit：`bd35c5a` "Update cylinder real shard 00018" 與 `0ae4426` "Bump data version"。cylinder 只有 shard 00018 變更；本地 2.0.0 的 184 個檔案和遠端 2.0.0 完全一致 | `$R/v2.0.1/step1.1/sha_compare.csv`、`remote_trees.json`、`local_old_sha256.txt` |
| 1.2 下載 | 下載 4 個檔案（749 MiB），其餘 181 個建 symlink；185 個檔案的 hash 全部驗證通過 | `$R/v2.0.1/step1.2_fetch.log`、`data_v2.0.1/fetch_manifest.json` |
| 1.3 Split | 所有 index 和 params JSON 逐位元相同。real：train 9063 / val 4820 / test 4827。**val 和 test 共用全部 92 個 sim，75% 的 test 樣本和 val 的時間窗重疊**；train 和 test 有 11% 的時間窗重疊。`3656.h5`：val 110 筆、test 86 筆、**train 0 筆** | `$R/v2.0.1/step1.3/split_analysis.json` |
| 2.1 完整性 | 92 條軌跡的形狀都是 (3990, 64, 128)，沒有 NaN 或 Inf；**只有 `3656.h5` 的統計量改變**（舊 u_mean 1.168 是全資料集最高，新 0.156） | `$R/v2.0.1/step2.1/real_stats_v2.0.{0,1}.csv`、`changed_traj_side_by_side.csv` |
| 2.2 差異性質 | 仿射換算：u/v 共用 a = 0.068381（b 分別是 0.0762 和 0.0018），vo 乘 400.000，殘差 2.6e-8；lag = 0，沒有翻轉或正負號翻轉；座標從像素改成公尺，t 不變 | `$R/v2.0.1/data_diff/3656_{u,v,vo}.png`、`3656_diff.json` |
| 2.3 Stage 1 驗證 | 原本的腳本不存在，改用自寫的 `scripts/stage1_checks.py`（D-008）。v/vo 的正負號慣例一致（corr 0.91，v 翻號後是 −0.05）；dx = 2.535 mm、dy = 2.729 mm（非等向）、dt = 5 ms；divergence floor div_ratio 約 0.86；所有無因次結論新舊版相同 | `$R/v2.0.1/step2.3/stage1_checks_v2.0.{0,1}.csv` |

---

## 4. 舊 checkpoint 重評（Phase 3）

### 4.1 新舊對照（checkpoint 用 2.0.0 val 選的點；RMSE / Rel L2）

| 模型 | 設定 | 官方：舊 → 新 | 重訓 s0：舊 → 新 |
|---|---|---|---|
| persistence | – | 0.0434 → 0.0364 / 0.182 → 0.181 | – |
| U-Net | numerical | 0.0758 → 0.0373 / 0.217 → 0.210 | 0.0756 → 0.0375 / 0.217 → 0.211 |
| U-Net | real | 0.0700 → 0.0118 / 0.070 → 0.061 | 0.0706 → 0.0122 / 0.072 → 0.063 |
| U-Net | finetune | 0.0632 → 0.0125 / 0.073 → 0.065 | 0.0616 → 0.0127 / 0.073 → 0.066 |
| DeepONet | numerical | 0.0863 → 0.0599 / 0.359 → 0.356 | 0.0867 → 0.0599 / 0.345 → 0.341 |
| DeepONet | real | 0.0713 → 0.0252 / 0.153 → 0.146 | 0.0731 → 0.0280 / 0.169 → 0.162 |
| DeepONet | finetune | 0.0661 → 0.0249 / 0.150 → 0.144 | 0.0669 → 0.0243 / 0.147 → 0.140 |
| Transolver | numerical | 0.1029 → 0.0571 / 0.323 → 0.315 | 0.1029 → 0.0566 / 0.332 → 0.324 |
| Transolver | real | 0.0978 → 0.0326 / 0.187 → 0.176 | 0.0962 → 0.0345 / 0.197 → 0.187 |
| Transolver | finetune | 0.0864 → 0.0288 / 0.171 → 0.161 | 0.0847 → 0.0306 / 0.183 → 0.174 |

- 全部 10 個指標（RMSE、MAE、Rel L2、R²、fRMSE 和 low/mid/high、FE、KE）的新、舊、變化 %：`$R/v2.0.1/reeval_summary.{csv,md}`
- 每個評估一個 JSON：`$R/v2.0.0/reeval/*.json`、`$R/v2.0.1/reeval/*.json`（檔名 `{model}_{setting}_{official|retrained_s0}.json`、`persistence.json`）
- Job 清單：`pi-lfm-code/jobs/phase3_main.json`；slurm log：`$R/slurm/p3-v2.0.{0,1}-g{1,2,3}-*`、`p3b-v2.0.1-g1-461185`
- 2.0.0 時期的原始 eval log（09-15/16）：`/work/b314513067/pi-lfm/runs/{model}/{model}_cylinder_eval/<ts>/eval.log`，彙整在 `/work/b314513067/pi-lfm/runs/RESULTS.md`；harness 在 2.0.0 上的結果和這些 log 一致到 5 位小數

### 4.2 Early-peak 檢查（Step 3.4）

| 模型 | numerical | real | finetune |
|---|---|---|---|
| U-Net | 4600 → 9800 | 6400 → 6800 | 2000 → 6800 |
| DeepONet | 300 → 1300 | 1500 → 4700 | 4700 → 5000 |
| Transolver | 100 → 1100 | **600 → 5000** | **500 → 5000** |

（最佳點：2.0.0 val → 2.0.1 val；seed 0 重訓 run 的 50 個中間 checkpoint 全部重評）

- **曲線圖**：`$R/v2.0.1/val_curves_summary.png`（舊 val 的存檔曲線 vs 新 val 的重評曲線，3×3）；表：`val_curves_summary.csv`
- 每個點的全部指標：`$R/v2.0.1/val_curves/{model}__{exp}__{ts}.json`（U-Net 那 3 條另有 `.json.partial`：前 47 個點是從 slurm log 取回的 6 位小數值，見 §7）
- 控制組（舊 val 重算 vs checkpoint 存檔，50 個點，最大相對差異 1.55e-7）：`$R/v2.0.0/val_curves_control/`
- slurm log：`$R/slurm/vc-*`、`vc2-*`

---

## 5. Baseline 重訓與測試（Phase 4）

### 5.1 重訓方式
- numerical **不重訓**（code 和資料都沒變），只在 2.0.1 val 上重新選點。
- real **seed 0 不重訓**：real train 資料逐位元相同，同一個 seed 會得到相同的權重軌跡，所以改為重新選點（D-017，**和任務字面要求不同**）；**seed 1、2 新訓練**。
- finetune **3 個 seed 全部新訓練**（包括 seed 0）。起點是 2.0.1 val 選出的 numerical 最佳點：U-Net `model_9800`、DeepONet `model_1300`、Transolver `model_1100`（D-021、D-022）。
- 用官方 `realpdebench.train`；config 只改了 `dataset_root`、`results_path`、`seed`、`exp_name`（加 `_s{seed}`）、`num_workers: 6`，finetune 另外改 `checkpoint_path`（D-018）。啟動腳本：`pi-lfm-code/slurm/train_baseline.sh`。

### 5.2 Test 結果（2.0.1 real test，N_ar = 1；mean ± std，ddof = 1）

| 模型 | 設定 | n | RMSE | MAE | Rel L2 | R² | fRMSE | FE | KE |
|---|---|---|---|---|---|---|---|---|---|
| persistence | – | – | 0.0364 | 0.0128 | 0.181 | 0.648 | 0.00450 | 47.6 | 3.6e-4 |
| U-Net | numerical | 1 | 0.0371 | 0.0225 | 0.209 | 0.634 | 0.00564 | 112 | 2.6e-4 |
| U-Net | real | 3 | 0.01209 ± 0.00004 | 0.00530 | 0.0620 ± 0.0003 | 0.961 | 0.00141 | 33.1 ± 1.7 | 1.30e-4 |
| U-Net | finetune | 3 | 0.01209 ± 0.00002 | 0.00522 | 0.0618 ± 0.0002 | 0.961 | 0.00140 | 34.6 ± 6.0 | 1.29e-4 |
| DeepONet | numerical | 1 | 0.0574 | 0.0373 | 0.321 | 0.125 | 0.00908 | 199 | 3.3e-4 |
| DeepONet | real | 1 | 0.0248 | 0.0131 | 0.143 | 0.837 | 0.00342 | 52.5 | 2.5e-4 |
| DeepONet | finetune | 1 | 0.0234 | 0.0122 | 0.135 | 0.855 | 0.00313 | 52.2 | 2.5e-4 |
| Transolver | numerical | 1 | 0.0545 | 0.0320 | 0.299 | 0.210 | 0.00875 | 149 | 3.4e-4 |
| Transolver | real | 3 | 0.0268 ± 0.0016 | 0.0130 | 0.145 ± 0.007 | 0.809 ± 0.022 | 0.00327 | 50.9 ± 2.5 | 2.6e-4 |
| Transolver | finetune | 3 | 0.0238 ± 0.0001 | 0.0114 | 0.130 ± 0.001 | 0.849 ± 0.002 | 0.00286 | 47.7 ± 0.3 | 2.4e-4 |

- 完整表（含 fRMSE low/mid/high 的 mean ± std）：`$R/v2.0.1/phase4_test_summary.md`
- 每個 checkpoint 一個 JSON：`$R/v2.0.1/phase4_test/{model}_{setting}_s{seed}.json`；job 清單：`pi-lfm-code/jobs/phase4_test.json`；slurm log：`$R/slurm/p4-test-475155.*`
- **學習曲線（所有 seed，2.0.1 val RMSE）**：`$R/v2.0.1/phase4_val_curves.png`；每個 run 的最佳點：`phase4_val_curves.csv`

### 5.3 每個 run 的 checkpoint、log、曲線位置

縮寫：`$O = /work/b314513067/pi-lfm/runs`（2.0.0 時期的 seed-0 run），`$B = $R/v2.0.1/baselines/runs`。每個 run 目錄裡都有 50 個 `model_XXXX.pth`、`training.log`、`events.out.tfevents.*`。

| 模型 | 設定 | seed | 用於 test 的 checkpoint（2.0.1 val 最佳點） | 曲線來源 | slurm log |
|---|---|---|---|---|---|
| U-Net | numerical | 0 | `$O/unet/unet_cylinder_numerical_False/2026-09-15_20-22-46/model_9800.pth` | `$R/v2.0.1/val_curves/unet__unet_cylinder_numerical_False__2026-09-15_20-22-46.json` | 原始訓練：`$O/slurm/cyl-unet-sim-389904.*` |
| U-Net | real | 0 | `$O/unet/unet_cylinder_real_False/2026-09-15_20-59-14/model_6800.pth` | `$R/v2.0.1/val_curves/unet__unet_cylinder_real_False__2026-09-15_20-59-14.json` | `$O/slurm/cyl-unet-real-390065.*` |
| U-Net | real | 1 | `$B/unet/unet_cylinder_s1_real_False/2026-09-29_19-51-54/model_6000.pth` | 存在 checkpoint 的 `val_losses`（同目錄 `training.log`） | `$R/slurm/p4-unet-real-s1-455169.*` |
| U-Net | real | 2 | `$B/unet/unet_cylinder_s2_real_False/2026-09-29_19-51-59/model_6600.pth` | 同上 | `$R/slurm/p4-unet-real-s2-455170.*` |
| U-Net | finetune | 0 | `$B/unet/unet_cylinder_s0_real_True/2026-09-30_14-57-58/model_6800.pth` | 同上 | `$R/slurm/p4-unet-ft-s0-461265.*` |
| U-Net | finetune | 1 | `$B/unet/unet_cylinder_s1_real_True/2026-09-30_14-57-57/model_6000.pth` | 同上 | `$R/slurm/p4-unet-ft-s1-461267.*` |
| U-Net | finetune | 2 | `$B/unet/unet_cylinder_s2_real_True/2026-09-30_14-57-51/model_5800.pth` | 同上 | `$R/slurm/p4-unet-ft-s2-461269.*` |
| DeepONet | numerical | 0 | `$O/deeponet/deeponet_cylinder_numerical_False/2026-09-15_20-21-44/model_1300.pth` | `$R/v2.0.1/val_curves/deeponet__deeponet_cylinder_numerical_False__2026-09-15_20-21-44.json` | `$O/slurm/cyl-deeponet-sim-389881.*` |
| DeepONet | real | 0 | `$O/deeponet/deeponet_cylinder_real_False/2026-09-15_20-22-41/model_4700.pth` | `$R/v2.0.1/val_curves/deeponet__deeponet_cylinder_real_False__2026-09-15_20-22-41.json` | `$O/slurm/cyl-deeponet-real-389902.*` |
| DeepONet | finetune | 0 | `$B/deeponet/deeponet_cylinder_s0_real_True/2026-09-30_14-57-51/model_5000.pth` | checkpoint 的 `val_losses` | `$R/slurm/p4-deeponet-ft-s0-461271.*` |
| Transolver | numerical | 0 | `$O/transolver/transolver_cylinder_numerical_False/2026-09-15_20-22-46/model_1100.pth` | `$R/v2.0.1/val_curves/transolver__transolver_cylinder_numerical_False__2026-09-15_20-22-46.json` | `$O/slurm/cyl-trainsolver-sim-389907.*` |
| Transolver | real | 0 | `$O/transolver/transolver_cylinder_real_False/2026-09-15_20-59-15/model_5000.pth` | `$R/v2.0.1/val_curves/transolver__transolver_cylinder_real_False__2026-09-15_20-59-15.json` | `$O/slurm/cyl-trainsolver-real-390066.*` |
| Transolver | real | 1 | `$B/transolver/transolver_cylinder_s1_real_False/2026-09-29_19-52-02/model_5000.pth` | checkpoint 的 `val_losses` | `$R/slurm/p4-trainsolver-real-s1-455171.*` |
| Transolver | real | 2 | `$B/transolver/transolver_cylinder_s2_real_False/2026-09-29_20-28-09/model_5000.pth` | 同上 | `$R/slurm/p4-trainsolver-real-s2-455776.*` |
| Transolver | finetune | 0 | `$B/transolver/transolver_cylinder_s0_real_True/2026-09-30_14-57-57/model_5000.pth` | 同上 | `$R/slurm/p4-trainsolver-ft-s0-461266.*` |
| Transolver | finetune | 1 | `$B/transolver/transolver_cylinder_s1_real_True/2026-09-30_14-57-58/model_5000.pth` | 同上 | `$R/slurm/p4-trainsolver-ft-s1-461268.*` |
| Transolver | finetune | 2 | `$B/transolver/transolver_cylinder_s2_real_True/2026-09-30_14-57-51/model_5000.pth` | 同上 | `$R/slurm/p4-trainsolver-ft-s2-461270.*` |

- 每個新 run 的實際 config（diff 印在 slurm .out 的開頭）：`$R/v2.0.1/baselines/configs/{unet,trainsolver,deeponet}_real[_ft]_s{seed}.yaml`
- 2.0.0 時期 seed-0 run 對應的 config：`pi-lfm-code/configs/cylinder/`（`*_nw6.yaml`、`*_finetune.yaml` 是當時啟動腳本產生的檔案）
- 官方 checkpoint 沒有中間 checkpoint，也沒有可以重新選點的曲線，只在 §4.1 做評估。

### 5.4 Update Ratio（只用於 finetune）

定義（RealPDEBench 官方指標，https://realpdebench.github.io/metrics/data-oriented/ ）：**Update Ratio = N₁/N₂**。RMSE₀ 是 real-world training 的最佳 RMSE；N₂ 和 N₁ 分別是 real training 和 finetuning 達到 RMSE₀ 所需的更新次數。< 1 代表 sim 預訓練減少了所需的更新次數。

官網沒有規定的細節，依 D-026 採用：**2.0.1 val RMSE**、官方評估間隔（num_update/50）、N₂ = real run 最佳點的 iteration、N₁ = finetune **第一次** val RMSE ≤ RMSE₀ 的 iteration、real 和 finetune **按 seed 配對**、沒達到就標 not reached。

| 模型 | seed | RMSE₀（val） | N₂ | N₁ | Update Ratio |
|---|---|---|---|---|---|
| U-Net | 0 | 0.012008 | 6800 | 6800 | 1.000 |
| U-Net | 1 | 0.012004 | 6000 | 5200 | 0.867 |
| U-Net | 2 | 0.011935 | 6600 | – | **not reached**（finetune 最佳 0.011971） |
| DeepONet | 0 | 0.024505 | 4700 | 2200 | 0.468 |
| Transolver | 0 | 0.026411 | 5000 | 1200 | 0.240 |
| Transolver | 1 | 0.025134 | 5000 | 2100 | 0.420 |
| Transolver | 2 | 0.028172 | 5000 | 600 | 0.120 |

- 結果：`$R/v2.0.1/phase4_update_ratio.csv`；腳本：`pi-lfm-code/scripts/update_ratio.py`（曲線來源和 §5.3 相同）。
- 解讀：和 §5.2 一致，sim 預訓練能幫 Transolver 和 DeepONet 大幅減少達到同樣表現所需的更新次數（0.12–0.47）；對 U-Net 幾乎沒有幫助（0.87–1.0，有一個 seed 沒達到）。
- **和論文不可直接比較**：用官方 checkpoint 存的 val 曲線，以任何一種常見算法都無法同時重現論文的 0.3636 / 0.5758 / 1.0（照字面的算法得到 0.030 / 0.515 / 0.125）。論文可能用了 test 曲線、不同的 run，或者某個沒公開的細節。

---

## 6. 偏離任務清單的地方（完整理由見 `DECISIONS.md`）

| # | 內容 |
|---|---|
| D-003 | 官方 repo 有一個先前就存在的本地修改（記憶體修正），維持現狀，沒有還原 |
| D-008 | Stage 1 的腳本和報告不存在，改為自寫驗證（使用者指示） |
| D-014 | 評估用自寫的 harness（逐行複製官方迴圈），沒有直接呼叫 `eval.py`；控制組確認兩者一致 |
| D-017 | real seed 0 沒有重訓，改為重新選點（train 資料沒變） |
| D-018 | baseline config 額外改了 `num_workers` 和 `exp_name`（不影響數值） |
| D-021 | finetune 的起點用我們重訓的 numerical（2.0.1 val 最佳點），沒有用官方 numerical |
| Step 3.1 順序 | 全精度控制組通過**之前**，就先送出了 Phase 3 的評估矩陣（控制組最後通過，結果不受影響） |

---

## 7. 執行異常與處理

| 異常 | 影響 | 處理 |
|---|---|---|
| 有些節點（hgpn003、hgpn146、hgpn111、hgpn143）會讓 `srun` 行程拿不到可用的 GPU：可能看不到 CUDA、CUDA 初始化失敗，或分到的 GPU 已經被其他行程佔滿 | 一個 Phase 3 job 靜默改用 CPU，4 小時後超時；第一次官方控制組在 CPU 上跑；3 條 U-Net val 曲線超時（47/50 點） | `slurm/gpu_guard.sh`：job 一開始先檢查 CUDA 和可用記憶體，失敗就排除該節點、原樣重新提交；harness 沒有 GPU 時直接報錯；val 曲線改成每個點都存檔、可以續跑，缺的點已補齊 |
| `$R/v2.0.1/reeval/unet_numerical_official.json` 是在 CPU 上算的 | 和 GPU 結果的差異約 3e-5 量級，不影響結論 | 保留不覆蓋（規則 3），metadata 的 host 是 hgpn003 |
| 叢集規定每張 GPU 最多 12 個 CPU；sbatch 偶爾會回暫時性錯誤 | 少數提交失敗或重複送出 | 改用 `scripts/submit_queue.sh` 排隊提交；重複的 job 已取消 |

---

## 8. 尚未完成（Phase 0–4 範圍）

- **Step 3.6**（N_ar = 10 與 MVPE）：可選項目，需要使用者另外批准，所以沒有做。
- **DeepONet 的多 seed**：任務說之後再補，目前只有 1 個 seed。
- **Step 4.3 其餘 baseline**：任務本來就只要求排程、不執行。排程見 `docs/reports/step_4_baselines.md`。
