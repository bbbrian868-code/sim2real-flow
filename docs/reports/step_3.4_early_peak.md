# Step 3.4 報告：Early-peak 現象檢查

- 範圍：9 個重訓 run（seed 0），每個 run 有 50 個中間 checkpoint，全部在 **2.0.1 real val**（4820 筆）上重新評估。評估方式照 `train.py:345-373`，指標在 CPU 上算，和官方相同。
- 指令：`scripts/val_curves.py`（每個 run 一個 slurm job）；圖和表：`scripts/plot_val_curves.py`
- 產出：`results/v2.0.1/val_curves/*.json`（每個點的全部 14 種指標）、`results/v2.0.1/val_curves_summary.{png,csv}`
- 控制組：同一個 harness 在 **2.0.0 val** 上重算 DeepONet numerical 的 50 個點，和 checkpoint 裡存的 `val_losses['rmse']` 相比，最大相對差異是 **1.55e-7**（`results/v2.0.0/val_curves_control/`）
- 注意：U-Net 3 條曲線的前 47 個點，是從 job log 取回的 6 位小數值（原始 job 在 4 小時上限超時，見 Phase 3 報告的「執行異常」），最後 3 個點是重算的。這不影響最佳點的判斷。

| 模型 | 設定 | 最佳點（2.0.0 val） | 最佳點（2.0.1 val） | 佔訓練預算 | 最佳 val RMSE（舊 → 新） |
|---|---|---|---|---|---|
| U-Net | numerical | 4600 | 9800 | 98% | 0.0831 → 0.0368 |
| U-Net | real | 6400 | 6800 | 68% | 0.0797 → 0.0120 |
| U-Net | finetune | 2000 | 6800 | 68% | 0.0694 → 0.0120 |
| DeepONet | numerical | 300 | 1300 | 26% | 0.0926 → 0.0570 |
| DeepONet | real | 1500 | 4700 | 94% | 0.0814 → 0.0245 |
| DeepONet | finetune | 4700 | 5000 | 100% | 0.0745 → 0.0241 |
| Transolver | numerical | 100 | 1100 | 22% | 0.1123 → 0.0539 |
| Transolver | real | **600** | **5000** | 100% | 0.1072 → 0.0264 |
| Transolver | finetune | **500** | **5000** | 100% | 0.0944 → 0.0238 |

**結論**
- **real 和 finetune 的 early-peak 現象在 2.0.1 上消失了。** 在新 val 上，這些曲線幾乎單調下降，最佳點落在預算的 68–100%。原本 Transolver 在 600 和 500 就見頂後回升，那是 val 裡 `3656.h5` 量級錯誤造成的假象：它在 val 的平方誤差裡權重過大，而且跟著訓練逐漸惡化。
- numerical 模型在新 val 上仍然很早就進入平台（DeepONet 和 Transolver 在預算的 22–26%，U-Net 大約 4000 步之後），這反映的是 sim 到 real 的落差。它們的新最佳點落在平台上，和相鄰點的差距在噪聲量級。
- 影響：finetune 的起點（numerical 的最佳點）改變了，所以 finetune 的 seed 0 必須重訓（D-022）；real 的 seed 0 改用新選出的點（D-017）。
