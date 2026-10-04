# Phase 6 報告：FM 正式實驗結果（Step 6.4）

資料集 cylinder，dataset 2.0.1，real 測試集，N_ar = 1。指標全部用官方 `eval_metrics` 計算：評估 u、v 兩個通道，4827 個樣本，metric_batch_size = 全部，所以 RMSE = √(全域平均平方誤差)。

完整表：`results/fm/phase6_summary/phase6_summary.md`（CSV：`phase6_main.csv`、`phase6_K_decomposition.csv`、`phase6_N_sweep.csv`、`phase6_update_ratio.csv`；舊版總覽圖：`phase6_val_curves.png`，已由下面的 `curves/` 取代）。
腳本：`scripts/summarize_phase6.py`。每個 run 一個 JSON：`results/fm/phase6_test/`，內含兩個 repo 的 commit、dataset_version、config 全文與 sha256、seed、checkpoint 路徑。

## 共同設定（每張表都適用）

- **方法**：基礎版 rectified flow（5.0 凍結設定）。**生成對象 = 完整 T_out 幀**（殘差版本尚未做，6.3 第 3 項留待下次）。
- **評估**：EMA 權重；checkpoint = K=1、N=10 的 val RMSE 最小點（536 筆 val 子集，D-011）；測試 N = 20 步 Euler，K = 1 與 K = 5，噪聲 seed 1234。
- **目標是否含 sim 加噪**：官方的乘性高斯噪聲（scale 0.1）同時加在輸入和目標上，而且只對 numerical 資料（Step 0.2）。所以 numerical 和 finetune（預訓練階段）是 **「目標含 sim 加噪」**；real 設定沒有 sim 資料，不受影響。
- **√2 理論差距（5.0 已知影響 (1)）**：如果模型完美學到 p(y|c)，單次採樣的 MSE 是條件均值的 2 倍，RMSE 差 √2 = 1.414 倍；K = 5 平均時差 √1.2 = 1.095 倍。U-Net baseline 是確定性回歸，理想上會逼近條件均值。所以 **FM 的 K = 1 數字天生比 baseline 吃虧**，實際觀測到的差距見後面的 K 拆解。
- **訓練預算**：numerical 和 real 都是 50k 步；finetune 從 numerical 最佳點的 EMA 權重接續，20k 步，lr × 0.3（D-023、D-024）。
- **大小**：U-Net-M 38.8M、U-Net-L 125.5M、DiT-S 9.9M、DiT-M 39.1M、DiT-L 119.4M（D-012、D-019；DiT 的 patch 是 2，偏離規格）。**U-Net-S 作廢，改用 U-Net-Sv2（9.34M），已於 10-02 重訓完成**（見異常 1）。

## 主表：M 檔（3 seeds，mean ± std）與對照

RMSE 括號內是相對同設定 U-Net baseline 的變化。

| 方法 | 設定 | 目標含 sim 加噪 | N | K | RMSE | Rel L2 | R² | fRMSE high | FE | KE | Update Ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| persistence | – | – | – | – | 0.0364 | 0.181 | 0.648 | 0.00325 | 47.6 | 3.6e-4 | – |
| U-Net baseline（確定性） | numerical | 是 | – | – | 0.0371（n=1） | 0.209 | 0.634 | 0.00300 | 112 | 2.6e-4 | – |
| FM U-Net-M | numerical | 是 | 20 | 1 | 0.0335 ± 0.0003（−10%） | 0.195 | 0.703 | 0.00278 | 113 | 4.2e-4 | – |
| FM U-Net-M | numerical | 是 | 20 | 5 | 0.0300 ± 0.0002（−19%） | 0.170 | 0.760 | 0.00273 | 111 | 2.7e-4 | – |
| FM DiT-M | numerical | 是 | 20 | 1 | 0.0391 ± 0.0002（+5%） | 0.217 | 0.594 | 0.00347 | 153 | 4.3e-4 | – |
| FM DiT-M | numerical | 是 | 20 | 5 | 0.0361 ± 0.0002（−3%） | 0.193 | 0.655 | 0.00339 | 152 | 3.2e-4 | – |
| U-Net baseline（確定性） | real | 否（無 sim） | – | – | 0.01209 ± 0.00004 | 0.0620 | 0.961 | 0.00124 | 33.1 | 1.30e-4 | – |
| FM U-Net-M | real | 否（無 sim） | 20 | 1 | 0.01473 ± 0.00012（+22%） | 0.0826 | 0.942 | 0.00142 | 48.7 | 1.57e-4 | – |
| FM U-Net-M | real | 否（無 sim） | 20 | 5 | 0.01350 ± 0.00004（+12%） | 0.0710 | 0.952 | 0.00138 | 41.1 | 1.38e-4 | – |
| FM DiT-M | real | 否（無 sim） | 20 | 1 | 0.01592 ± 0.00039（+32%） | 0.0831 | 0.933 | 0.00163 | 45.5 | 1.64e-4 | – |
| FM DiT-M | real | 否（無 sim） | 20 | 5 | 0.01331 ± 0.00024（+10%） | 0.0688 | 0.953 | 0.00137 | 38.7 | 1.42e-4 | – |
| U-Net baseline（確定性） | finetune | 是 | – | – | 0.01209 ± 0.00002 | 0.0618 | 0.961 | 0.00124 | 34.6 | 1.29e-4 | 1.00 / 0.87 / not reached |
| FM U-Net-M | finetune | 是 | 20 | 1 | 0.01681 ± 0.00013（+39%） | 0.0976 | 0.925 | 0.00160 | 48.2 | 1.89e-4 | not reached ×3 |
| FM U-Net-M | finetune | 是 | 20 | 5 | 0.01361 ± 0.00005（+13%） | 0.0730 | 0.951 | 0.00137 | 43.1 | 1.42e-4 | not reached ×3 |
| FM DiT-M | finetune | 是 | 20 | 1 | 0.01733 ± 0.00021（+43%） | 0.0917 | 0.920 | 0.00165 | 40.8 | 1.79e-4 | not reached ×3 |
| FM DiT-M | finetune | 是 | 20 | 5 | 0.01436 ± 0.00015（+19%） | 0.0753 | 0.945 | 0.00146 | 34.8 | 1.54e-4 | not reached ×3 |

## 大小比較（seed 0；M 檔是 3 seeds 的平均）

RMSE，K = 1 / K = 5：

| backbone | 大小 | numerical | real | finetune |
|---|---|---|---|---|
| U-Net | Sv2（取代 S，D-028） | 0.0365 / 0.0321 | 0.0170 / 0.0142 | 0.0186 / 0.0141 |
| U-Net | M | 0.0335 / 0.0300 | 0.0147 / 0.0135 | 0.0168 / 0.0136 |
| U-Net | L | 0.0307 / 0.0275 | **0.0128 / 0.0125** | 0.0143 / 0.0131 |
| DiT | S | 0.0407 / 0.0363 | 0.0175 / 0.0146 | 0.0264 / 0.0194 ※ |
| DiT | M | 0.0391 / 0.0361 | 0.0159 / 0.0133 | 0.0173 / 0.0144 |
| DiT | L | 0.0382 / 0.0360 | 0.0158 / 0.0132 | 0.0164 / 0.0136 |
| U-Net baseline | – | 0.0371 | 0.0121 | 0.0121 |

※ DiT-S finetune 在第 4000 步發散，best.pt 在第 2000 步（D-029）。

## 主要觀察

1. **real 和 finetune 設定下，FM 在所有指標上都比確定性 U-Net baseline 差**，K = 5 也一樣。最接近的是 U-Net-L real：K = 1 差 +5.9%，K = 5 差 +3.6%。頻譜指標也沒有優勢：fRMSE high 和 KE 都比 baseline 大；FE 最好的情況（U-Net-L real K = 5 是 33.2，DiT-L finetune K = 5 是 33.4）也只是和 baseline（33.1 / 34.6）打平。所以在這個評估協議下，**看不到「生成模型保留高頻細節」的好處**。
2. **numerical（sim → real zero-shot）設定下，FM U-Net 反而比 baseline 好**：M 檔 K = 1 好 10%，L 檔好 17%。不過 baseline 本身就比 persistence 還差（0.0371 vs 0.0364），而且 FM 的 FE 仍然高達 100 以上。DiT 和 baseline 差不多。
3. **K 拆解**：用 MSE_K = A + V/K 從 K = 1 和 K = 5 解出 A（K→∞ 時樣本平均的誤差）和 V（模型的採樣變異）。完美模型應該是 V/A = 1。實際觀測的 V/A 是：U-Net-M real 0.25、U-Net-L real 0.06、DiT-M real 0.61；K1/K∞ 的比值是 1.03–1.28，都小於 √2。
   - 解讀：**FM 的落後主要不是採樣變異造成的**。K→∞ 的誤差 A 本身就比 baseline 大：U-Net-M real 的 √A = 0.0132，比 baseline 0.0121 高 9%；U-Net-L 是 0.0125，高 3%。也就是說，FM 的「條件均值」就已經比回歸模型差，然後還要再付採樣變異的代價。
   - V/A < 1 可能有兩個原因：模型的分布太窄，或 A 裡模型偏差的比例很大。光憑 K = 1、5 無法區分這兩者。
4. **finetune 沒有幫助 FM（但用的是非論文設定，見 D-030）**：所有 FM finetune 都比同 backbone 的 real 差，Update Ratio 全部是 not reached。U-Net-M real 在第 50k 步仍在進步，finetune 第 20k 步也仍在進步。原因很可能是 finetune 的預算（20k 步、lr × 0.3，D-024）比 real 的 50k 步少很多，而不是預訓練有害。這個比較目前不公平，需要你決定是否要讓 finetune 跑到相同預算（見待決事項）。
5. **6.3 第 4 項（只對條件加噪）**：numerical 設定明顯變好（U-Net-M 0.0337 → 0.0308，−8.6%；DiT-M 0.0393 → 0.0365，−7.2%；都是 seed 0、K = 1）。也就是說，**目標含 sim 加噪確實讓 FM 學去生成那種噪聲**，符合 5.0 已知影響 (2) 的預期。但 finetune 之後差異就消失了：U-Net-M 0.0172 vs 0.0170，DiT-M 0.0169 vs 0.0171。
6. **6.3 第 1 項（N 掃描）**：RMSE 隨 N 增加而**微幅變差**。以 seed 0、K = 1 為例：DiT-M real 在 N = 10/20/50 分別是 0.0153 / 0.0158 / 0.0162，U-Net-M real 幾乎不變（0.01487 / 0.01484 / 0.01488）。這符合「步數少 → Euler 誤差偏向平滑 → 比較接近條件均值」的解釋，所以 RMSE 低不代表樣本品質好。正式表格維持 N = 20。

## 訓練曲線與 eval 曲線

圖檔：`results/fm/phase6_summary/curves/*.png`，由 `scripts/plot_phase6_curves.py` 產生。資料全部來自每個 run 目錄裡的 `log.jsonl`，不需要另外重算。

**每張分組圖的讀法**：三欄是 numerical / real / finetune，三列分別是：
1. **training loss**：FM 速度場 MSE，`log.jsonl` 裡帶 `loss` 的行，每 50 步記一次單一 batch 的值。細線是原始值，粗線是 1000 步的移動平均。
2. **grad norm**：clip 1.0 之前的梯度範數（`grad_norm`），1000 步移動平均。用來看 DiT 的發散（D-029）。
3. **eval curve**：val RMSE，`log.jsonl` 裡帶 `val` 的行，每 2000 步一次。用 536 筆 val 子集、N = 10、K = 1、EMA 權重。這就是選 checkpoint 用的曲線，**★ 是 best.pt**，也就是測試表用的那個 checkpoint。

顏色代表 seed（藍 = 0、橘 = 1、綠 = 2），所有圖都一樣。val 是 536 筆子集、N = 10，所以它的數值和測試表（4827 筆、N = 20）不能直接比。

### 圖與檔案的對應

`$P = /work/b314513067/pi-lfm/results/fm`

| 圖 | 對應主表 / 段落 | 使用的 run（每個都在 `$P/phase6/<run>/`，含 `log.jsonl`、`run_meta.json`、`best.pt`） | 測試結果 JSON |
|---|---|---|---|
| `curves/unet_M.png` | 主表 FM U-Net-M | `unet_M_{numerical_s0, numerical_s1, numerical_s2}`、`unet_M_{real_s0b, real_s1, real_s2}`、`unet_M_{finetune_s0, finetune_s1, finetune_s2}` | `$P/phase6_test/<run>_N20_K{1,5}.json` |
| `curves/dit_M.png` | 主表 FM DiT-M | `dit_M_{numerical_s0b, numerical_s1, numerical_s2}`、`dit_M_{real_s0b, real_s1, real_s2}`、`dit_M_{finetune_s0, finetune_s1, finetune_s2}` | 同上 |
| `curves/unet_L.png` | 大小比較 U-Net-L | `unet_L_{numerical, real, finetune}_s0` | 同上 |
| `curves/dit_L.png` | 大小比較 DiT-L | `dit_L_{numerical, real, finetune}_s0` | 同上 |
| `curves/dit_S.png` | 大小比較 DiT-S | `dit_S_{numerical, real, finetune}_s0` | 同上 |
| `curves/unet_Sv2.png` | 大小比較 U-Net-Sv2（D-028） | `unet_Sv2_{numerical, real, finetune}_s0` | `$P/phase6_test/<run>_N20_K{1,5}.json` |
| `curves/unet_M_condnoise.png` | 觀察 5（6.3 第 4 項） | `unet_M_condnoise_{numerical, finetune}_s0` | 同上 |
| `curves/dit_M_condnoise.png` | 觀察 5（6.3 第 4 項） | `dit_M_condnoise_{numerical, finetune}_s0` | 同上 |
| `curves/unet_S.png` | 作廢（D-028），僅留存 | `unet_S_{numerical, real, finetune}_s0` | 同上（INVALID 區塊） |
| `curves/size_overview.png` | 大小比較 | 上面各組的 seed 0，只畫 eval curve | – |

說明：
- seed 0 的 M 檔 real（兩個 backbone）和 DiT-M numerical 是 `*_s0b`：原本的 `*_s0` 送件失敗，目錄裡只有空的 log，所以重送到 `_s0b`。`unet_M_numerical_s0` 本身就是完整的 run。
- finetune 的起點寫在各 run 的 `run_meta.json` 的 `init_from`。例如 `dit_M_finetune_s0` 的起點是 `dit_M_numerical_s0b/best.pt`。
- K 拆解、N 掃描、Update Ratio 用的 JSON 分別在 `phase6_test/`、`phase6_ablation/N_sweep/`，以及上面這些 `log.jsonl`。

### U-Net-M

![U-Net-M curves](../../../pi-lfm/results/fm/phase6_summary/curves/unet_M.png)

- 三個 seed 幾乎重疊，grad norm 平穩下降，沒有不穩定。
- **numerical**：val 在約 22k 步最低（★），之後慢慢變差，但 training loss 還在降。也就是模擬資料訓練越久，對 real val 反而越不利，和 baseline 的 early-peak 現象一致。
- **real**：val 到 50k 步仍在下降，★ 就在最後一步，表示預算還不夠（異常 4）。
- **finetune**：起點的 val 已經在 0.018 左右，20k 步結束時還在下降，★ 也在最後一步。這是「20k 步、lr × 0.3 不夠」（D-030）的直接證據。

### DiT-M

![DiT-M curves](../../../pi-lfm/results/fm/phase6_summary/curves/dit_M.png)

- **numerical**：val 在第 2000 步最低（★），之後一路變差，從 0.038 升到 0.044，training loss 卻幾乎不動。和 U-Net 比起來，DiT 更快 overfit 到模擬資料。
- **real**：約 27k 步時三個 seed 同時發散：grad norm 從 0.1 跳到 10⁴–10⁸，training loss 回到約 1，val RMSE 跳到約 0.13。★ 都在發散前（20k–26k 步），所以測試數字有效（D-029）。
- **finetune**：grad norm 平穩，沒有發散，20k 步結束時 val 仍在下降。

### 大小比較（seed 0，只畫 eval curve）

![size overview](../../../pi-lfm/results/fm/phase6_summary/curves/size_overview.png)

- U-Net：real 和 finetune 都是 L < M < Sv2，L 從第一個評估點就比較好。
- DiT：real 在發散前 M 和 L 幾乎重疊，S 稍高。發散的時間點不固定：S 在 numerical 第 26k 步、finetune 第 4k 步；L 在 real 第 18k 步。DiT-S finetune 的 ★ 因此停在第 2000 步。

### 其他大小與消融

![U-Net-L curves](../../../pi-lfm/results/fm/phase6_summary/curves/unet_L.png)

![DiT-L curves](../../../pi-lfm/results/fm/phase6_summary/curves/dit_L.png)

![DiT-S curves](../../../pi-lfm/results/fm/phase6_summary/curves/dit_S.png)

DiT-S finetune 從一開始 grad norm 就持續上升，從 0.5 升到約 2500 步時的 10²，之後發散，所以 ★ 停在第 2000 步。numerical 則是在 24k–26k 步突然發散。

![U-Net-Sv2 curves](../../../pi-lfm/results/fm/phase6_summary/curves/unet_Sv2.png)

U-Net-Sv2 已訓練完成（10-02）。舊 U-Net-S 的 training loss 停在 0.4–0.5，Sv2 正常下降，寬度不足的問題已經排除。測試 RMSE 依大小排序為 Sv2 > M > L，例如 real K=1 是 0.0170 / 0.0147 / 0.0128。和其他大小一樣，finetune（20k 步、lr × 0.3）比 real 差，Update Ratio 為 not reached。

![U-Net-M condnoise curves](../../../pi-lfm/results/fm/phase6_summary/curves/unet_M_condnoise.png)

![DiT-M condnoise curves](../../../pi-lfm/results/fm/phase6_summary/curves/dit_M_condnoise.png)

只對條件加噪的版本沒有 real 欄：real 資料本來就沒有模擬噪聲，所以 real 和基礎版完全相同。

**numerical 的 training loss 直接印證 5.0 的已知影響 (2)**：U-Net-M 只對條件加噪時，loss 降到約 0.02；基礎版（目標也加噪）停在約 0.18。DiT-M 也一樣，基礎版約 0.16。多出來的這一截，就是模型在學著生成目標上的乘性模擬噪聲，而這部分無法從輸入預測。

![U-Net-S curves (INVALID)](../../../pi-lfm/results/fm/phase6_summary/curves/unet_S.png)

作廢的 U-Net-S（D-028）：三個設定的 training loss 都停在 0.4–0.5，val 也降不下來。這張圖只是留存。

## 場的視覺化（單一 real 測試樣本）

腳本：`scripts/field_viz.py`。`compute` 階段要用 GPU（job 487418，約 2 分鐘），`plot` 階段只用 CPU。產出放在 `results/fm/phase6_summary/field_viz/`：

| 檔案 | 內容 |
|---|---|
| `field_real.png` | 圖 1：real-world training |
| `field_zeroshot.png` | 圖 2：zero-shot（只用模擬資料訓練） |
| `fields.npz` | 畫圖用的所有場：GT、兩個 baseline、FM 的 5 個樣本 ×（real, numerical），最後一幀，物理單位 |
| `meta.json` | 樣本編號、選樣規則、所有 checkpoint 路徑、兩個 repo 的 commit |
| `sample_rmse.json` | 圖上標的單幀 RMSE |

**樣本怎麼選的**：用 U-Net baseline（real, seed 0）算整個 2.0.1 real 測試集每個樣本的 RMSE（u、v，全部 20 幀），取**中位數**那一個：test #3456，sim `6656.h5`，t0 = 1890，RMSE 0.00877。這是一個典型樣本，不是挑過的。兩張圖用同一個樣本，都畫**最後一個預測幀**（第 20/20 幀），單位是物理單位（反正規化後）。

**使用的模型**

| 欄 | checkpoint |
|---|---|
| U-Net baseline (real) | `runs/unet/unet_cylinder_real_False/2026-09-15_20-59-14/model_6800.pth`（Phase 4 的 real seed 0） |
| U-Net baseline (simulated) | `runs/unet/unet_cylinder_numerical_False/2026-09-15_20-22-46/model_9800.pth`（Phase 4 的 numerical seed 0） |
| FM U-Net-M (real) | `results/fm/phase6/unet_M_real_s0b/best.pt`（EMA） |
| FM U-Net-M (simulated) | `results/fm/phase6/unet_M_numerical_s0/best.pt`（EMA） |

FM 的設定是 N = 20 Euler、噪聲 seed 1234。K=1 的樣本就是 5 個樣本中的第一個，所以和 `FMPredictor(K=1, seed=1234)` 對單一樣本的輸出完全相同。K=5 是 5 個樣本的平均；std 是這 5 個樣本的標準差（ddof = 1）。

### 圖 1：real-world training

![field real](../../../pi-lfm/results/fm/phase6_summary/field_viz/field_real.png)

版面：每個變數兩列。上列是場本身，同一列共用色階，以 GT 的 0.5–99.5 百分位為範圍；v 用以 0 為中心的發散色階。下列是 |error|。FM std 放在誤差列，和 |error| **用同一個色階**，這樣採樣的變異可以直接和誤差比大小。

- 這一幀的 RMSE：U-Net baseline u 0.0065 / v 0.0153；FM K=1 是 0.0093 / 0.0210；FM K=5 是 0.0074 / 0.0199。排序和主表一致。
- **FM 的誤差和 baseline 的誤差位置大致相同**，都集中在尾流的渦結構上。也就是說，兩者在同樣的地方犯錯，FM 並沒有把渦的細節補得更好。
- **FM 的單次樣本有逐像素的雜點**，自由流區也有。這部分 K=5 平均後大多消失，對應 K 拆解裡的 V。
- **FM std（RMS 0.0069）比誤差小得多**，而且分布在尾流整區，不是集中在誤差最大的地方。和 K 拆解 V/A < 1 一致：模型的採樣分散度不足以涵蓋它實際的誤差。
- **邊界效應（只看了這一個樣本）**：FM K=1 在最外圈 1 像素的平均誤差是內部的 2.4 倍（0.0186 vs 0.0078），K=5 是 1.4 倍。往內一格就恢復正常，例如第 1 列是 0.004。U-Net baseline 沒有這個現象（0.95 倍）。可能和卷積在邊界的 zero padding 有關，要在整個測試集上確認後才能下結論。

### 圖 2：zero-shot（只用模擬資料訓練）

![field zero-shot](../../../pi-lfm/results/fm/phase6_summary/field_viz/field_zeroshot.png)

- 這一幀的 RMSE(u)：U-Net baseline 0.0408，FM K=5 是 0.0258，和主表 numerical 列「FM 比 baseline 好」的方向一致。
- **兩者錯的方式不同**：U-Net baseline 的場很平滑，但渦的位置和形狀都錯了，誤差是大塊的結構性誤差。FM 的大尺度結構比較接近（尾流的寬度和位置），但整張圖蓋著一層雜點。
- **這層雜點就是 FM 學去生成的模擬噪聲**：numerical 訓練時，官方的乘性高斯噪聲（0.1 × 場）也加在目標上（5.0 已知影響 (2)）。5 個樣本平均後雜點還是很明顯，自由流的 u 約 0.2，0.1 倍就是約 0.02，平均 5 次後約 0.009。這和「只對條件加噪」的版本在 numerical 上好 7–9% 一致。

## 異常與與預期不符之處

1. **U-Net-S（base_ch 40）作廢，已由 Sv2 取代（D-028）**。U-Net 第一層把 y_t（每像素 60 維）和 cond 壓到 base_ch 個通道，40 < 60 時 y_t 的逐像素噪聲傳不過去。三個設定的訓練 loss 都卡在 0.4–0.5（M 檔降到 0.02），測試 RMSE 約 0.085，**比 persistence 還差**；K 拆解的 V/A = 34，代表模型輸出的幾乎只是噪聲。舊結果另列在完整表的「INVALID」區塊。新設定 **U-Net-Sv2** 是 base_ch 72、channel_mult 1-1-2-2、9.34M，job 487143（numerical）和 487144（real），finetune 和 eval 會由 `jobs/queue/phase6_sv2.q` 自動送出。跑完重新執行 `summarize_phase6.py`，表格就會更新。
2. **DiT 訓練不穩定（D-029）**：6 個 DiT run 在訓練中途發散，梯度範數漲到 10³–10⁴，loss 回到約 1。包括 DiT-M real 全部 3 個 seed（27–30k 步）、DiT-L real（18–28k 步）、DiT-S numerical（26k 步）、DiT-S finetune（4k 步）。測試用的 best.pt 都在發散前，所以數字有效，只有 DiT-S finetune 等於只訓練了 2000 步。U-Net 沒有這個問題。
3. **FM 的 numerical 最佳點都很早**：DiT 是 2000–4000 步；U-Net-M 是 22k 步，之後 val 變差。這和 baseline 的 early-peak 現象（Step 3.4）一致：sim 訓練越久，對 real val 越不利。
4. **U-Net-M real 在第 50k 步仍在進步**（best = last），表示 50k 步的預算對 U-Net-M 還不夠；U-Net-L real 在第 34k 步就到最佳了。

## 需要你決定的事

1. **DiT 的穩定性**：要不要修？選項有 (a) QK-norm（backbone 內部的改動）、(b) 降低 lr，例如 5e-5、(c) 不修，維持現狀照實報告。我傾向 (a)，但要重訓所有 DiT run，而且會改到凍結的訓練設定，所以沒有自行執行。
2. ~~finetune 預算~~ → 已決定照論文的設定，**先不重訓**（D-030）。論文和官方程式碼的 finetune 是：步數、lr、batch、scheduler 都和 real 相同，只載入模型權重，optimizer 重新建立。對 FM 來說，要從 20k 步、lr × 0.3 改成 **50k 步、lr 1e-4**。**本報告中的 FM finetune 結果和 Update Ratio 都是「非論文設定」（D-024）**，不能直接拿來和論文或 baseline 的 finetune 比較。
3. **6.3 第 3 項（殘差目標）**：照你的指示留待下次。從結果來看，FM 和 baseline 的差距主要在條件均值的誤差 A，殘差目標正好可能改善這一點，所以建議優先做這一項。

DiT 穩定性（第 1 項）和殘差目標（第 3 項）：使用者 10-02 指示留到下次。

## 下一步（尚未執行）

- ~~U-Net-Sv2 的 3 個 run 和 eval~~：10-02 完成，彙整表、曲線和大小比較表都已更新。
