# Phase 3 報告：舊 checkpoint 重新評估（Step 3.1 / 3.2 / 3.3 / 3.5）

協議固定為 real test、N_ar = 1、test_mode = all、metric_batch_size = 整個測試集（4827）、c = 2（只算 u、v），和 Step 0.2 一致。
所有數字都經由 `pilfm/official_eval.py` 計算（逐行複製 `eval.py:290-352`，指標呼叫官方的 `eval_metrics`），見 D-014。

## Step 3.1 控制組

| 比對 | 結果 |
|---|---|
| harness vs 官方 `eval.py`（`scripts/official_eval_capture.py`，在執行時攔截 `eval_metrics` 的回傳值，不改原始碼），官方 U-Net numerical，2.0.0 資料，**兩者都在 GPU 上** | RMSE、Rel L2 **逐位元相同**（0.07575702667）；13 個指標的最大相對差異是 **1.56e-7**（freq_error）→ **通過**（門檻 < 1e-6） |
| harness vs 09-15 的舊 eval.log（5 位小數），18 個 checkpoint，2.0.0 | 全部在 log 精度內一致，例如 0.075757 / 0.07576、0.086340 / 0.08634、0.102926 / 0.10293 |

**異常（已處理）**：第一次官方 capture（job 455069）被分到 `25a-hgpn146`，那個 `srun` 行程看不到 GPU，官方 `eval.py` 就靜默改用 CPU。這讓最大相對差異變成 2.8e-4，而且跑了 2 小時。在 GPU 上重跑（job 461218）後通過。同樣的節點問題也讓 Phase 3 的一個 job 在 CPU 上跑到超時（見下面的「執行異常」）。

**流程偏差（誠實記錄）**：規則要求控制組通過後才能做後續步驟。實際上我在 harness 和 5 位小數的舊紀錄一致後，就先送出了 Phase 3 的評估矩陣，和全精度控制組平行跑。全精度控制組最後通過，所以結果不受影響，但執行順序不符合規則。

## Step 3.2 Persistence baseline

預測方式：把最後一幀輸入（反正規化後）複製成 20 幀輸出，再用 target 的統計量正規化，其餘走同一條評估路徑。

| | RMSE | MAE | Rel L2 | R² | fRMSE | low | mid | high | FE | KE |
|---|---|---|---|---|---|---|---|---|---|---|
| 2.0.0 | 0.04340 | 0.01403 | 0.1822 | 0.8784 | 0.00516 | 0.00436 | 0.00662 | 0.00403 | 51.29 | 0.00051 |
| 2.0.1 | 0.03640 | 0.01283 | 0.1807 | 0.6482 | 0.00450 | 0.00391 | 0.00588 | 0.00325 | 47.61 | 0.00036 |

檔案：`results/v2.0.{0,1}/reeval/persistence.json`

## Step 3.3 / 3.5 完整評估矩陣與新舊對照

- 完整表（10 種指標 × 新、舊、變化 %）：`results/v2.0.1/reeval_summary.{csv,md}`；每個評估一個 JSON，附 metadata：`results/v2.0.{0,1}/reeval/*.json`
- 重訓的 checkpoint 用的是 **2.0.0 val 選出的最佳點**（Step 0.1）。在 2.0.1 val 上重新選點的結果見 Step 3.4。

| 模型 | 設定 | 來源 | RMSE 舊→新 | Δ% | Rel L2 舊→新 | Δ% | R² 舊→新 |
|---|---|---|---|---|---|---|---|
| persistence | – | – | 0.0434 → 0.0364 | −16 | 0.182 → 0.181 | −1 | 0.878 → 0.648 |
| U-Net | numerical | 官方 | 0.0758 → 0.0373 | −51 | 0.217 → 0.210 | −3 | 0.630 → 0.631 |
| U-Net | numerical | 重訓 | 0.0756 → 0.0375 | −50 | 0.217 → 0.211 | −3 | 0.631 → 0.628 |
| U-Net | real | 官方 | 0.0700 → **0.0118** | −83 | 0.070 → 0.061 | −13 | 0.684 → 0.963 |
| U-Net | real | 重訓 | 0.0706 → 0.0122 | −83 | 0.072 → 0.063 | −13 | 0.678 → 0.960 |
| U-Net | finetune | 官方 | 0.0632 → 0.0125 | −80 | 0.073 → 0.065 | −11 | 0.743 → 0.959 |
| U-Net | finetune | 重訓 | 0.0616 → 0.0127 | −79 | 0.073 → 0.066 | −10 | 0.755 → 0.957 |
| DeepONet | numerical | 官方 | 0.0863 → 0.0599 | −31 | 0.359 → 0.356 | −1 | 0.519 → 0.048 |
| DeepONet | numerical | 重訓 | 0.0867 → 0.0599 | −31 | 0.345 → 0.341 | −1 | 0.514 → 0.046 |
| DeepONet | real | 官方 | 0.0713 → 0.0252 | −65 | 0.153 → 0.146 | −5 | 0.672 → 0.831 |
| DeepONet | real | 重訓 | 0.0731 → 0.0280 | −62 | 0.169 → 0.162 | −4 | 0.655 → 0.791 |
| DeepONet | finetune | 官方 | 0.0661 → 0.0249 | −62 | 0.150 → 0.144 | −4 | 0.718 → 0.835 |
| DeepONet | finetune | 重訓 | 0.0669 → 0.0243 | −64 | 0.147 → 0.140 | −5 | 0.711 → 0.843 |
| Transolver | numerical | 官方 | 0.1029 → 0.0571 | −45 | 0.323 → 0.315 | −3 | 0.316 → 0.135 |
| Transolver | numerical | 重訓 | 0.1029 → 0.0566 | −45 | 0.332 → 0.324 | −2 | 0.317 → 0.151 |
| Transolver | real | 官方 | 0.0978 → 0.0326 | −67 | 0.187 → 0.176 | −6 | 0.382 → 0.718 |
| Transolver | real | 重訓 | 0.0962 → 0.0345 | −64 | 0.197 → 0.187 | −5 | 0.402 → 0.685 |
| Transolver | finetune | 官方 | 0.0864 → 0.0288 | −67 | 0.171 → 0.161 | −5 | 0.518 → 0.780 |
| Transolver | finetune | 重訓 | 0.0847 → 0.0306 | −64 | 0.183 → 0.174 | −5 | 0.536 → 0.751 |

## 解讀

1. **2.0.0 的 RMSE、MAE、R² 被一條軌跡主導。** `3656.h5` 只佔 test 的 1.78%（86/4827），但它的數值量級約是其他軌跡的 15 倍（Step 2.2：新 = 0.0684 × 舊 + b）。模型對它的平方誤差佔了總 SSE 的大半。修正之後，real/finetune 模型的 RMSE 下降 62–83%，numerical 模型下降 31–51%。
2. **Rel L2 幾乎不變（−1% ~ −13%）**，因為它是逐樣本相對於 ‖target‖ 做正規化，量級錯誤不會放大 3656 的權重。論文 Table 1 的 Rel L2 欄因此大致仍然成立，RMSE 欄則不成立。
3. **排名的變化**：在 2.0.0 上，persistence 的 RMSE（0.0434）**比所有學習型模型都好**；在 2.0.1 上，所有 real/finetune 模型都贏過 persistence，而且 U-Net 領先的幅度大幅擴大（0.012 vs DeepONet 0.025、Transolver 0.029）。numerical 訓練的模型在 RMSE 上仍然輸給 persistence。
4. **R² 的變化方向不一致**：numerical 模型的 R² 大跌（DeepONet 0.52 → 0.05），因為 target 的總變異原本大部分來自 3656，分母縮小之後，泛化差的模型就露出原形。real/finetune 模型的 R² 大幅上升（U-Net 到 0.96）。
5. **fRMSE 和 FE 的排名反轉**：在 2.0.0 上，persistence 的 fRMSE（0.0052）和 FE（51）**比所有學習型模型都好**（U-Net real 分別是 0.0103 和 142）。到了 2.0.1，real/finetune 模型全部反超（U-Net real 分別是 0.0014 和 35；DeepONet finetune 是 0.0033 和 51），只有 numerical 模型仍然輸給 persistence。KE 也一樣，在 2.0.1 上 U-Net real 的 0.00013 優於 persistence 的 0.00036。
6. **官方 vs 重訓**：U-Net 兩者差距在 ±4% 內；DeepONet real 和 Transolver real/finetune 的重訓版比官方版差 6–11%。

## 執行異常

- `p3-v2.0.1-g1`（job 455143）被分到 `25a-hgpn003`，`srun` 行程看不到 CUDA，harness 靜默改用 CPU。`unet_numerical_official` 花了 9750 秒（在 GPU 上約 280 秒），之後 4 小時超時。修正（commit `Fail fast…`）：沒有 CUDA 就直接失敗，並排除 hgpn003 和 hgpn146。缺少的 5 個評估由 job 461185 在 GPU 上補跑。
- 在 CPU 上算出的 `v2.0.1/reeval/unet_numerical_official.json` 保留下來，沒有覆蓋（規則 3），其 metadata 的 host 是 hgpn003。它和 GPU 結果的差異在 3e-5 量級（參考 Step 3.1），不影響任何結論。
