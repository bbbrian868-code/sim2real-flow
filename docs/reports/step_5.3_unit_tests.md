# Step 5.3 報告：單元測試

- 指令：`python fm/tests/test_fm.py --data`（在計算節點上以 slurm job 執行：455338 是第一版，461222 是包含快取路徑的版本）。結果：**ALL PASSED**。
- (d) 另外寫成 `fm/tests/check_eval_path.py`，因為要在 GPU 上跑完整的測試集。

| 項目 | 結果 |
|---|---|
| (a) 形狀 | U-Net-S（有、無 bottleneck attention）、DiT-S patch 2/4/8，batch 1/2/5，輸入輸出形狀都正確；`unfold(fold(x)) == x` 逐位元相同；折疊後的第 t·C + c 個通道 = 第 t 幀的第 c 個欄位；`FMPredictor` 輸出 `(B, 20, 64, 128, 3)` |
| (b) Sampler | 解析速度場 v = (y* − x)/(1 − t)，Euler N = 1/5/20/50 步，最後一步和 y* 的最大誤差是 2.4e-7 / 2.4e-7 / 1.2e-7 / 1.2e-7（float32 精度內） |
| (c) 初始 loss | 最後一層 zero-init 時，各 backbone 的輸出全為 0，FM loss 和 mean((y1 − y0)²) 的相對差異 < 1e-6。**實際資料**（各取 256 筆 train、正規化後）：**numerical：E[(y1−y0)²] = E[y1²] + 1 = 2.013**（E[y1²]，u/v/p = 1.036 / 0.923 / 1.081；含 sim 加噪與遮蔽）；**real：1.251**（u/v/p = 0.224 / 0.514 / 0.016）。實際訓練的初始 loss（smoke 和 overfit log 的第 1 個記錄）和這兩個值一致：numerical 約 2.07，real 約 1.25 |
| (d) 評估路徑一致性 | **待跑**（job 送出時碰到權限檢查無回應）。方法：用假的速度場 v = (p(cond) − y)/(1 − t)，其中 p 是 persistence 預測，經過 `FMPredictor` 和官方鏡像的評估迴圈，比對 Step 3.2 的 persistence 指標 |
| (e) Checkpoint round-trip | model、EMA、AdamW 狀態存檔再讀回後，輸出逐位元相同；再多做一步 optimizer 更新，結果仍逐位元相同 |
| (f) 資料等價性（附加） | `FastCylinderHFDataset` 的 zero-copy 路徑和快取路徑，在相同 RNG 狀態下，和官方 `CylinderHFDataset` 的輸出**逐位元相同**；real train、real val、numerical train 各測 4 筆，包含遮蔽和乘法型加噪的亂數抽取 |

**sim 加噪**：官方的噪聲同時加在 input 和 target 上（Step 0.2 §3）。所以 FM 在 numerical 和 finetune 的 sim 階段學到的目標**含有 sim 加噪**；Step 6.3 第 4 項（只對條件加噪）必須做。
