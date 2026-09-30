# Step 5.4 報告：單一 batch 過擬合測試

- 設定：real train 的第一個 batch（seed 0，64 筆，real 資料沒有擴增的隨機性），固定重複使用；bf16；AdamW，lr 1e-4，cosine；1 張 H200。
- 指令：`slurm/fm.sh --config fm/configs/cylinder/{unet_M,dit_M}.yaml --train-data-type real --fixed-batch --no-val --iters N [--warmup 100]`
- 評估：訓練結束後，用 raw 和 EMA 權重在同一個 batch 上做 N=20 步 Euler 採樣（固定噪聲 seed），計算正規化空間的 Rel L2，對 64 筆取平均。
- 產出：`results/fm/step5.4_overfit/`（第一版）、`results/fm/step5.4_overfit_w100/`（修正版），每個目錄有 `log.jsonl`、`overfit_eval.json`、`diag_model.json`

| run | 排程 | loss @500 / 1000 / 2000 / 10000 | N=20 Rel L2（EMA） |
|---|---|---|---|
| U-Net-M | warmup 1000，2000 步（原排程） | – / 0.690 / 0.485 / – | 1.42 |
| DiT-M，patch 4 | warmup 1000，2000 步 | – / 0.591 / 0.542 / – | 1.51 |
| U-Net-M | warmup 100，2000 步 | 0.706 / 0.546 / 0.452 / – | 1.35 |
| U-Net-M | warmup 100，10000 步 | 0.696 / 0.514 / 0.318 / **0.069** | **0.385** |
| DiT-M，patch 4 | warmup 100，10000 步 | 0.576 / 0.559 / 0.533 / **0.501（平台）** | 1.47 |
| **DiT-M，patch 2** | warmup 100，2000 步 | 0.130 / 0.105 / **0.074** / – | **0.313** |

初始 loss 約 1.25 = E[y1²] + 1（real，見 Step 5.3 (c)）。

**發現 1：原本的排程不適合做這個測試。** warmup 1000 加上在 2000 步內 cosine 到 0，有效學習量太少。改成 warmup 100 之後才看得出趨勢（D-020）。

**發現 2：patch 4 的 DiT 有結構性的 loss 下限，約 0.5。** 用 `fm/tests/diag_overfit.py` 按 t 分段診斷（2000 步的 checkpoint）：

| t | 0 | 0.3 | 0.5 | 0.7 | 0.9 | 0.99 |
|---|---|---|---|---|---|---|
| DiT-M p4 的 FM loss | 0.528 | 0.509 | 0.508 | 0.512 | 0.559 | 0.883 |
| U-Net-M 的 FM loss | 0.423 | 0.421 | 0.422 | 0.426 | 0.454 | 0.820 |

在 t=0 時 y_t = y0，目標是 y1 − y_t，只需要恆等映射加上記住的 y1。DiT p4 在這裡的 loss 仍有 0.53，而且跑到 10000 步都不再下降。原因是每個 token 的 y_t 有 4²×60 = 960 維，大於 hidden 512，patch embedding 無法把 y_t 的逐像素白噪聲傳到輸出，下限約為 1 − 512/960 ≈ 0.47，和觀測相符。規格的預設 patch 8（3840 維）更嚴重。改成 **patch 2**（240 維）後，2000 步 loss 就到 0.074（**D-019，偏離規格**）。

**發現 3：U-Net 學得比較慢，但會收斂。** 2000 步時仍比「只輸出 −y_t」的 0.265 差，10000 步時到 0.069。

**Rel L2 為什麼停在 0.3–0.4：** real 的 target 用 numerical 的統計量正規化後，RMS 只有 0.51（u 0.47、v 0.74、p 0.13），而起點噪聲是 N(0, 1)。在 t → 1 時，要從 y_t 推回 y0 需要除以 (1 − t)，速度場的殘差會被放大，最後留下的噪聲相對於 ‖y1‖ 就不小（diag：t=0.99 時 loss 約 0.82）。這是規格 5.0（高斯起點、完整幀目標）在 real 資料上的特性，**不是 bug**，會在 Phase 6 的結果裡一併說明。

**結論**：訓練、損失、採樣的路徑都正確。兩個 backbone 在修正後都能過擬合（U-Net-M 需要大約 10k 步，DiT-M patch 2 大約 2k 步）。
