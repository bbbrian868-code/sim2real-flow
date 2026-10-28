# Task: DiT 預測目標（x vs v）× 模型寬度 實驗

> 本檔是使用者在 2026-10-26 貼給 Claude Code 的任務清單原文（原檔名 `task_dit_xpred_width.md`），原樣存檔以便接續。
> 之後的決定（例如 patch、paradigm、散度評估延後）以根目錄 `DECISIONS.md` 為準，會覆蓋本檔的部分內容。

> 背景與文獻依據見 `dit_xpred_plan.md`（JiT: Li & He, CVPR 2026；RAE: Zheng et al., ICLR 2026）。
> 本檔是給 Claude Code 的執行清單。

---

## 目標

在 RealPDEBench cylinder 上，比較 DiT flow-matching backbone 的兩種輸出參數化（v-prediction / x-prediction）在四種寬度下的表現，並記錄計算成本。

主實驗網格（8 runs）：

| | width 384 | width 512 | width 768 | width 1024 |
|---|---|---|---|---|
| v-pred | R1 | R2 | R3 | R4 |
| x-pred | R5 | R6 | R7 | R8 |

除了 `pred` 與 `width`（及隨之變動的 heads 數），其餘全部固定。

---

## 全域規則（每一步都適用）

1. **每完成一個 Step 就停下回報**，等我回覆「繼續」才進下一步。
2. **未經明確同意，不得啟動任何正式訓練或推論**（smoke test 僅限 Step 4 規定的範圍）。
3. **不得修改既有預設行為**：所有新功能以 config 開關加入，預設值必須重現舊行為。
4. **遇到文件或程式碼語意不明時，不得自行假設**，列出選項並回報。
5. **發現架構或設定前後不一致時，直接標出來**，不要自行做局部修補。
6. 不得使用 test split 做任何 checkpoint 選擇或超參數決定。
7. 只回報數字與觀察到的現象，不要替結果下研究結論。
8. **維護決策紀錄 `DECISIONS.md`**（放在 repo 根目錄，見下節）：
   - Step 0 開始前先建立此檔，並把下方「已確定的決策」D1–D10 原樣寫入。
   - 之後每次我在 STOP 點做出決定（包含選擇方案、否決方案、核准或不核准執行），都要新增一筆，寫明決定內容、理由、被否決的替代方案。
   - 理由必須是我給的理由；我沒說明理由時，填「理由：待補」並在回報中提醒我，**不得自行代填理由**。
   - 已記錄的條目不得刪改；若決定被推翻，新增一筆並在「取代」欄註明被取代的編號。
   - 每步回報的最後附上本步新增的決策編號。

---

## 決策紀錄格式與已確定的決策

（D1–D10 的全文已原樣寫入 repo 根目錄的 `DECISIONS.md`，這裡不重複。）

---

## Step 0　盤點現有 DiT FM 設定（唯讀，不改任何檔案）

找到目前 DiT flow-matching 模型、訓練腳本、config、評估腳本，填寫下表並回報：

| 項目 | 值 | 出處（檔案:行號） |
|---|---|---|
| depth（DiT block 數） | | |
| hidden size / heads / head dim | | |
| patch size $p$ | | |
| 輸入通道組成（條件幀數 × 通道 + 雜訊目標通道） | | |
| 輸出通道數 | | |
| **輸入 token 維度**（$C_\text{in} \cdot p^2$） | | |
| **輸出 token 維度**（$C_\text{out} \cdot p^2$） | | |
| 條件注入方式（channel concat / adaLN / cross-attn） | | |
| 時間方向慣例（$t=0$ 是雜訊還是資料？） | | |
| 目前網路輸出的是什麼（v / x / ε） | | |
| loss 定義與權重 | | |
| 時間取樣分布（uniform / logit-normal + 參數） | | |
| ODE solver 與步數（NFE） | | |
| 混合精度（fp16 / bf16） | | |
| 是否有 QK-norm | | |
| optimizer / LR / schedule / batch size / 總 iters | | |
| 資料版本（應為 v2.0.1）與訓練 paradigm | | |
| checkpoint 選擇規則 | | |
| 評估腳本與參數（`N_ar`、`test_mask_prob`、`metric_batch_size`） | | |

另外回報：

- 先前 DiT 訓練的 log 若還在，請描述症狀：loss 是否卡在高平台？有沒有 NaN / spike？$N_{ar}=1$ 的指標為何？
- **特別確認上次提到的「960」對應的是輸入還是輸出 token 維度。**

⛔ STOP — 回報後等待。

---

## Step 1　加入 `pred` 開關（x / v），兩者共用同一個 v-space loss

依 Step 0 確認的時間慣例實作。以下以 $t=0$ 雜訊、$t=1$ 資料為例；若 codebase 相反，整套對稱翻轉，並在回報中說明怎麼翻的。

```
z_t      = t * x + (1 - t) * eps
denom    = clamp_min(1 - t, t_clip)          # t_clip 預設 0.05
v_target = (x - z_t) / denom                 # t < 1 - t_clip 時等於 x - eps

pred="v":  v_pred = net(z_t, t, cond)
pred="x":  v_pred = (net(z_t, t, cond) - z_t) / denom

loss = mse(v_pred, v_target)
```

取樣時，`pred="x"` 每步先算 `v = (net(z,t,cond) - z) / clamp_min(1-t, t_clip)` 再走原本的 ODE solver。

需新增的 config 欄位：`pred: {"v","x"}`（預設 `"v"`）、`t_clip`（預設 0.05）。

驗證（全部要附輸出）：

1. **向後相容**：`pred="v"` 時，固定 seed 下第一個 batch 的 loss 與修改前完全相同（bit-exact，或列出差異來源）。
2. **target 一致性**：隨機抽 $t < 0.95$，檢查 `v_target` 與 `x - eps` 的最大絕對誤差。
3. **Oracle 測試**：把網路輸出直接換成真實 $x$（x-pred）或真實 $v$（v-pred），loss 應約等於 0；並用 oracle 跑一次取樣，確認能還原 $x$。
4. 列出所有被修改的檔案與 diff 摘要。

⛔ STOP

---

## Step 2　寬度參數化與成本表

- 以 config 支援 width ∈ {384, 512, 768, 1024}。
- **固定 head dim = 64**，所以 heads = 6 / 8 / 12 / 16。若原設定的 head dim 不是 64，先回報，不要自行改。
- depth、patch size、條件注入、MLP ratio 等與 Step 0 的原設定完全相同。
- LR 沿用原設定（已知寬度改變可能讓最佳 LR 偏移，這點先記為已知限制，不調整）。

對 8 個 config 各回報：

| run | pred | width | heads | 參數量 | GFLOPs / forward | 單次 forward 時間（ms, batch=1, bf16, H100） |
|---|---|---|---|---|---|---|

- FLOPs 用 `torch.utils.flop_counter.FlopCounterMode` 或 fvcore，註明使用哪一個。
- 時間用 CUDA event 計時：warm-up 20 次，量 100 次取中位數。

⛔ STOP

---

## Step 3　訓練期診斷 logging

加入以下 logging（以 config 開關控制，預設關閉）：

1. **分桶 loss**：每 N iters，把 v-space loss 依 $t$ 分成 5 桶（[0,0.2), …, [0.8,1.0]）各記平均。
2. **單步去噪誤差**：固定一個 val batch 與固定的 $t \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$、固定 noise seed，記錄 $\|\hat x - x\|^2 / \|x\|^2$。其中 $\hat x$ 在 x-pred 時是網路輸出，在 v-pred 時由 $\hat x = z_t + (1-t)\,v_\theta$ 換算。
3. **max attention logit**（每層取最大值，記全網路最大）。
4. **grad norm**。
5. **NaN / Inf 偵測**：出現即記錄 iteration 並存 checkpoint。

用任一 config 跑 50 iters 確認 logging 正常、沒有明顯 overhead（回報加 logging 前後的 it/s）。

⛔ STOP

---

## Step 4　Smoke test（僅此步允許短跑）

8 個 config 各在 1 張 GPU 上跑 **200 iters**，回報：

| run | 是否 NaN | 峰值 GPU 記憶體 | it/s | 推估完整訓練所需時間 | 200 iter 時的 loss |
|---|---|---|---|---|---|

並確認 checkpoint 存讀正常、log 檔路徑與命名能區分 8 個 run。

⛔ STOP — **等待我明確同意後才能進 Step 5。**

---

## Step 5　正式訓練（需事先同意）

- 8 個 run 平行，每張 H100 跑一個。
- 所有 run 使用相同 seed、相同資料順序、相同總 iters。
- 依 Step 0 回報的 checkpoint 規則存檔；另外每隔固定 iters 存一份，供事後分析。
- **不得用 test split 選 checkpoint。** 若既有規則用到 test，先回報，等我決定。
- 訓練中若任一 run 出現 NaN，不要自動重啟，記錄後回報。

完成後回報：各 run 最終 loss、訓練時間、是否有異常，以及 Step 3 的診斷曲線圖（同一張圖疊 v-pred 與 x-pred，按 width 分子圖）。

⛔ STOP

---

## Step 6　評估（需事先同意才能跑推論）

**評估協定必須與既有 baselines（U-Net / DeepONet / Transolver）完全相同**：同一個 eval 腳本，同樣的 `N_ar`、`test_mask_prob`、`metric_batch_size`，同一個 test set。若 FM 需要修改 eval 腳本才能跑，先回報修改方案。

每個 run：

1. **單步（$N_{ar}=1$）**：Rel L2、R²、KE、FE。每個測試樣本生成 $K=8$ 個樣本，以 ensemble mean 計算指標，另報 per-pixel std 的平均。
2. **Rollout（$N_{ar}=10$）**：每一步的 Rel L2 曲線。
3. **散度殘差**：只在計算物理量時套用 `v ← −v`，用實際格距 dx = 1.2639 mm、dy = 1.3551 mm（非等向）。回報 interior 中位數，並與資料本身的 floor（≈ 0.13）並列。
4. **能譜**：預測場與真實場的徑向能譜比較圖（取同一組樣本）。
5. **推論時間**：CUDA event 量測
   - 每幀時間 = 單次 forward × NFE（附 NFE 值）
   - 一條 $N_{ar}=10$ rollout 的總時間
   - $K=8$ ensemble 的總時間

輸出：

- `results/dit_xpred_width.csv`（每 run 一列，含 Step 2 的成本欄位）
- 圖 1：Rel L2（$N_{ar}=1$）vs 參數量，v-pred 與 x-pred 兩條線
- 圖 2：$N_{ar}=10$ rollout 誤差曲線，8 條線
- 圖 3：準確度 vs 每幀推論時間散點圖，並放入既有 baselines 的點（若有它們的計時資料；沒有就先回報）

⛔ STOP

---

## Step 7（可選，需事先同意）瓶頸 patch embedding

只在 Step 6 完成且我同意後執行。

- 設定：x-pred、width 512。
- 輸入投影改為兩層線性：`in_dim → d' → 512`，$d' \in \{64, 128\}$。
- 以 config 開關加入，預設關閉。
- 訓練與評估流程同 Step 5–6。

⛔ STOP

---

## 每步回報格式

```
### Step N 回報
- 完成項目：
- 修改的檔案：
- 驗證結果（附數字 / 輸出）：
- 發現的不一致或疑點：
- 需要我決定的事項：
- 本步新增至 DECISIONS.md 的條目：
```
