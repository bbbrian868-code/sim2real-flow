# Phase 4 報告：baseline 重訓

## Step 4.1 重訓原則（實際執行方式）
- **numerical 不重訓**：Step 0.3 確認上游 code 沒有更新，Step 1.1 確認 numerical 資料逐位元相同。不過它們的 checkpoint **改用 2.0.1 val 重新選點**（Step 3.4）：U-Net 9800、DeepONet 1300、Transolver 1100。
- **real seed 0 沒有重訓，改為沿用既有 run 並重新選點**（D-017，**和任務字面要求不同**）：2.0.1 的 real **train** 資料逐位元相同（`3656.h5` 只出現在 val/test），用同一個 seed 跑官方流程會得到相同的權重軌跡，唯一的差別在 checkpoint 選擇；既有 run 的 50 個中間 checkpoint 已經在 2.0.1 val 上全部重評過了。
- **finetune 的 3 個 seed 全部重訓，包括 seed 0**：numerical 的最佳點改變了，finetune 的起點也跟著不同（D-021、D-022）。起點是我們重訓的 numerical 依 2.0.1 val 選出的最佳點。
- 用官方 `realpdebench.train`，不改原始碼。config 和官方的差異只有 `dataset_root`、`results_path`、`seed`、`exp_name` 加上 `_s{seed}`、`num_workers: 6`，finetune 另外改 `checkpoint_path`（D-018）。腳本：`slurm/train_baseline.sh`，每個 run 的 config 放在 `results/v2.0.1/baselines/configs/`。
- val 曲線：官方每 num_update/50 步記錄一次，存在每個 checkpoint 的 `val_losses` 裡（14 種指標 × 50 點）；選點準則沿用官方的 val RMSE 最小值。

## Step 4.2 執行結果

| 模型 | 設定 | seeds | 最佳點（2.0.1 val） | 訓練時間 |
|---|---|---|---|---|
| U-Net | real | s0（沿用）、s1、s2 | 6800 / 6000 / 6600 | 約 3.8 h |
| U-Net | finetune | s0、s1、s2 | 6800 / 6000 / 5800 | 約 4.1 h |
| Transolver | real | s0（沿用）、s1、s2 | 5000 / 5000 / 5000 | 約 2.6 h |
| Transolver | finetune | s0、s1、s2 | 5000 / 5000 / 5000 | 約 2.5 h |
| DeepONet | real | s0（沿用） | 4700 | – |
| DeepONet | finetune | s0 | 5000 | 約 2.7 h |

全部在 H200 上執行，每個 run 用 2 張 GPU（多的那張只是為了 cgroup 記憶體，訓練只用 cuda:0），而且都確認 log 有 `Start training on cuda:0`。產出：`results/v2.0.1/baselines/runs/`。

## Test 結果（2.0.1，real test，N_ar = 1，mean ± std，ddof = 1）

完整表：`results/v2.0.1/phase4_test_summary.md`；每個 checkpoint 一個 JSON：`results/v2.0.1/phase4_test/`；job 清單：`jobs/phase4_test.json`

| 模型 | 設定 | n | RMSE | Rel L2 | R² | fRMSE | FE | KE |
|---|---|---|---|---|---|---|---|---|
| persistence | – | – | 0.0364 | 0.181 | 0.648 | 0.00450 | 47.6 | 3.6e-4 |
| U-Net | numerical | 1 | 0.0371 | 0.209 | 0.634 | 0.00564 | 112 | 2.6e-4 |
| U-Net | real | 3 | **0.01209 ± 0.00004** | 0.0620 ± 0.0003 | 0.961 | 0.00141 | 33.1 ± 1.7 | 1.30e-4 |
| U-Net | finetune | 3 | **0.01209 ± 0.00002** | 0.0618 ± 0.0002 | 0.961 | 0.00140 | 34.6 ± 6.0 | 1.29e-4 |
| DeepONet | numerical | 1 | 0.0574 | 0.321 | 0.125 | 0.00908 | 199 | 3.3e-4 |
| DeepONet | real | 1 | 0.0248 | 0.143 | 0.837 | 0.00342 | 52.5 | 2.5e-4 |
| DeepONet | finetune | 1 | 0.0234 | 0.135 | 0.855 | 0.00313 | 52.2 | 2.5e-4 |
| Transolver | numerical | 1 | 0.0545 | 0.299 | 0.210 | 0.00875 | 149 | 3.4e-4 |
| Transolver | real | 3 | 0.0268 ± 0.0016 | 0.145 ± 0.007 | 0.809 ± 0.022 | 0.00327 | 50.9 ± 2.5 | 2.6e-4 |
| Transolver | finetune | 3 | 0.0238 ± 0.0001 | 0.130 ± 0.001 | 0.849 ± 0.002 | 0.00286 | 47.7 ± 0.3 | 2.4e-4 |

**觀察**
- **U-Net 的 finetune 和 real 沒有差別**（RMSE 都是 0.0121，差距在 1 個 std 內）：sim 預訓練對 U-Net 沒有幫助。
- **Transolver 的 finetune 比 real 好 11%**（0.0238 vs 0.0268），而且 seed 之間的變異從 ±0.0016 降到 ±0.0001。DeepONet 只有 1 個 seed，finetune 也略好（−6%）。
- 在 2.0.1 val 上選點後，numerical 模型（U-Net 0.0371、DeepONet 0.0574、Transolver 0.0545）和 Step 3.3 用 2.0.0 val 選點的結果（0.0375、0.0599、0.0566）差距不大，仍然輸給 persistence（0.0364）或和它相當。
- 所有 real/finetune 模型在 RMSE、fRMSE、KE 上都贏過 persistence；**Transolver 和 DeepONet 的 FE（約 48–53）和 persistence 的 47.6 差不多**。

## Step 4.3 其餘 baseline 的排程（本任務不執行）

順序依任務清單。估計時間以 Nano4 H200、官方 config 為準，在第一次 smoke 前**都還沒驗證過**。

| 順序 | 模型 | 官方 config | 需要的設定 | 注意事項 |
|---|---|---|---|---|
| 1 | DMD | `configs/cylinder/dmd.yaml` | numerical / real / finetune | 不需要訓練（`load_model.py` 的 DMD 分支直接擬合），只需要評估 |
| 2 | FNO | `fno.yaml` | 同上 × 3 seeds（real/finetune） | |
| 3 | WDNO | `wdno.yaml` | 同上 | 屬於擴散類模型；評估要採樣，成本比較高 |
| 4 | DPOT-S | `dpot_s.yaml` | 同上 | 需要下載預訓練權重（`utils/dpot_ckpts_dl.py`） |
| 5 | CNO | `cno.yaml` | 同上 | 需要 `setuptools<81`（`pkg_resources`，見 `runs/RESULTS.md` §6）；**使用者已指示不需要 CNO（D-004），要做的話需重新確認** |
| 6 | MWT | `mwt.yaml` | 同上 | |
| 7 | GK-Transformer | `galerkin_transformer.yaml` | 同上 | |
| 8 | DPOT-L | `dpot_l.yaml` | 同上 | 需要預訓練權重；模型大，可能需要多卡 |

所有模型共用 `slurm/train_baseline.sh`（每個 run 有自己的 config，並有 GPU guard）和 `scripts/eval_ckpts.py`（評估），finetune 的起點沿用 D-021 的原則。

## Update Ratio（finetune）

RealPDEBench 官方指標（https://realpdebench.github.io/metrics/data-oriented/ ）：Update Ratio = N₁/N₂。RMSE₀ 是 real-world training 的最佳 RMSE；N₂ 和 N₁ 分別是 real training 和 finetuning 達到 RMSE₀ 所需的更新次數。官網沒有規定的細節依 **D-026**：2.0.1 val RMSE、官方評估間隔、N₁ = finetune 第一次達到 RMSE₀ 的 iteration、按 seed 配對。

| 模型 | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| U-Net | 1.000（6800/6800） | 0.867（5200/6000） | not reached（finetune 最佳 0.011971 > RMSE₀ 0.011935） |
| DeepONet | 0.468（2200/4700） | – | – |
| Transolver | 0.240（1200/5000） | 0.420（2100/5000） | 0.120（600/5000） |

結果：`results/v2.0.1/phase4_update_ratio.csv`；腳本：`scripts/update_ratio.py`。**和論文的數值不可直接比較**：用官方 checkpoint 的 val 曲線，以任何常見算法都無法重現論文的 0.3636 / 0.5758 / 1.0（見 D-026）。

## 補充實驗：numerical 的 checkpoint 改用模擬訓練 loss 選（10-04，D-031）

完整記錄見 [exp_sim_trainloss_selection.md](exp_sim_trainloss_selection.md)。上面 Test 結果的 numerical 列用的是官方的 real val 選點；在嚴格的 zero-shot 定義下不該用 real 資料，所以補做「用模擬訓練 loss 最小選點」，seed 0，2.0.1 real test：

| 模型 | real val 選點（上表） | 模擬訓練 loss 選點 | ΔRMSE |
|---|---|---|---|
| U-Net | 9800 步，RMSE 0.03711 | 8400 步，RMSE 0.03714 | +0.08% |
| DeepONet | 1300 步，RMSE 0.05740 | 4900 步，RMSE 0.05986 | +4.3% |
| Transolver | 1100 步，RMSE 0.05454 | 4900 步，RMSE 0.05864 | +7.5% |

- 模擬訓練 loss 一路降到最後，所以這種選法幾乎等於「訓練到底」；改用 5 倍視窗或直接取最後一個，結果都差不多。
- U-Net 不受影響；DeepONet、Transolver 變差，因為它們在 real val 上很早就到最低點（early-peak）。
- real val 選點的數字應視為上限：val 和 test 高度重疊（Step 1.3），而且 Transolver 的 val 曲線雜訊很大。

## 尚未完成
- DeepONet 的多 seed：任務說「之後再補 seed」，目前只有 1 個。
