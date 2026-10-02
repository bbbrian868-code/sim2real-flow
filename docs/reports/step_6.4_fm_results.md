# Phase 6 報告：FM 正式實驗結果（Step 6.4）

資料集 cylinder，dataset 2.0.1，real 測試集，N_ar = 1。指標全部用官方 `eval_metrics` 計算：評估 u、v 兩個通道，4827 個樣本，metric_batch_size = 全部，所以 RMSE = √(全域平均平方誤差)。

完整表：`results/fm/phase6_summary/phase6_summary.md`（CSV：`phase6_main.csv`、`phase6_K_decomposition.csv`、`phase6_N_sweep.csv`、`phase6_update_ratio.csv`；圖：`phase6_val_curves.png`）。
腳本：`scripts/summarize_phase6.py`。每個 run 一個 JSON：`results/fm/phase6_test/`，內含兩個 repo 的 commit、dataset_version、config 全文與 sha256、seed、checkpoint 路徑。

## 共同設定（每張表都適用）

- **方法**：基礎版 rectified flow（5.0 凍結設定）。**生成對象 = 完整 T_out 幀**（殘差版本尚未做，6.3 第 3 項留待下次）。
- **評估**：EMA 權重；checkpoint = K=1、N=10 的 val RMSE 最小點（536 筆 val 子集，D-011）；測試 N = 20 步 Euler，K = 1 與 K = 5，噪聲 seed 1234。
- **目標是否含 sim 加噪**：官方的乘性高斯噪聲（scale 0.1）同時加在輸入和目標上，而且只對 numerical 資料（Step 0.2）。所以 numerical 和 finetune（預訓練階段）是 **「目標含 sim 加噪」**；real 設定沒有 sim 資料，不受影響。
- **√2 理論差距（5.0 已知影響 (1)）**：如果模型完美學到 p(y|c)，單次採樣的 MSE 是條件均值的 2 倍，RMSE 差 √2 = 1.414 倍；K = 5 平均時差 √1.2 = 1.095 倍。U-Net baseline 是確定性回歸，理想上會逼近條件均值。所以 **FM 的 K = 1 數字天生比 baseline 吃虧**，實際觀測到的差距見後面的 K 拆解。
- **訓練預算**：numerical 和 real 都是 50k 步；finetune 從 numerical 最佳點的 EMA 權重接續，20k 步，lr × 0.3（D-023、D-024）。
- **大小**：U-Net-M 38.8M、U-Net-L 125.5M、DiT-S 9.9M、DiT-M 39.1M、DiT-L 119.4M（D-012、D-019；DiT 的 patch 是 2，偏離規格）。**U-Net-S 作廢，Sv2 重訓中**（見異常 1）。

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
| U-Net | S | 作廢（D-028） | 作廢 | 作廢 |
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
4. **finetune 沒有幫助 FM**：所有 FM finetune 都比同 backbone 的 real 差，Update Ratio 全部是 not reached。U-Net-M real 在第 50k 步仍在進步，finetune 第 20k 步也仍在進步。原因很可能是 finetune 的預算（20k 步、lr × 0.3，D-024）比 real 的 50k 步少很多，而不是預訓練有害。這個比較目前不公平，需要你決定是否要讓 finetune 跑到相同預算（見待決事項）。
5. **6.3 第 4 項（只對條件加噪）**：numerical 設定明顯變好（U-Net-M 0.0337 → 0.0308，−8.6%；DiT-M 0.0393 → 0.0365，−7.2%；都是 seed 0、K = 1）。也就是說，**目標含 sim 加噪確實讓 FM 學去生成那種噪聲**，符合 5.0 已知影響 (2) 的預期。但 finetune 之後差異就消失了：U-Net-M 0.0172 vs 0.0170，DiT-M 0.0169 vs 0.0171。
6. **6.3 第 1 項（N 掃描）**：RMSE 隨 N 增加而**微幅變差**。以 seed 0、K = 1 為例：DiT-M real 在 N = 10/20/50 分別是 0.0153 / 0.0158 / 0.0162，U-Net-M real 幾乎不變（0.01487 / 0.01484 / 0.01488）。這符合「步數少 → Euler 誤差偏向平滑 → 比較接近條件均值」的解釋，所以 RMSE 低不代表樣本品質好。正式表格維持 N = 20。

## 異常與與預期不符之處

1. **U-Net-S（base_ch 40）作廢，Sv2 重訓中（D-028）**。U-Net 第一層把 y_t（每像素 60 維）和 cond 壓到 base_ch 個通道，40 < 60 時 y_t 的逐像素噪聲傳不過去。三個設定的訓練 loss 都卡在 0.4–0.5（M 檔降到 0.02），測試 RMSE 約 0.085，**比 persistence 還差**；K 拆解的 V/A = 34，代表模型輸出的幾乎只是噪聲。舊結果另列在完整表的「INVALID」區塊。新設定 **U-Net-Sv2** 是 base_ch 72、channel_mult 1-1-2-2、9.34M，job 487143（numerical）和 487144（real），finetune 和 eval 會由 `jobs/queue/phase6_sv2.q` 自動送出。跑完重新執行 `summarize_phase6.py`，表格就會更新。
2. **DiT 訓練不穩定（D-029）**：6 個 DiT run 在訓練中途發散，梯度範數漲到 10³–10⁴，loss 回到約 1。包括 DiT-M real 全部 3 個 seed（27–30k 步）、DiT-L real（18–28k 步）、DiT-S numerical（26k 步）、DiT-S finetune（4k 步）。測試用的 best.pt 都在發散前，所以數字有效，只有 DiT-S finetune 等於只訓練了 2000 步。U-Net 沒有這個問題。
3. **FM 的 numerical 最佳點都很早**：DiT 是 2000–4000 步；U-Net-M 是 22k 步，之後 val 變差。這和 baseline 的 early-peak 現象（Step 3.4）一致：sim 訓練越久，對 real val 越不利。
4. **U-Net-M real 在第 50k 步仍在進步**（best = last），表示 50k 步的預算對 U-Net-M 還不夠；U-Net-L real 在第 34k 步就到最佳了。

## 需要你決定的事

1. **DiT 的穩定性**：要不要修？選項有 (a) QK-norm（backbone 內部的改動）、(b) 降低 lr，例如 5e-5、(c) 不修，維持現狀照實報告。我傾向 (a)，但要重訓所有 DiT run，而且會改到凍結的訓練設定，所以沒有自行執行。
2. **finetune 預算**：要不要讓 FM finetune 和 real 用相同的 50k 步、lr 不縮放，讓「預訓練是否有幫助」的比較變得公平？
3. **6.3 第 3 項（殘差目標）**：照你的指示留待下次。從結果來看，FM 和 baseline 的差距主要在條件均值的誤差 A，殘差目標正好可能改善這一點，所以建議優先做這一項。

## 下一步（尚未執行）

- U-Net-Sv2 的 3 個 run 和 eval 跑完後，重新產生彙整表，並補上這份報告的大小比較表。
