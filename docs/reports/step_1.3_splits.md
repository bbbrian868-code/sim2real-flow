# Step 1.3 報告：Split 比對

- 指令：`python scripts/split_analysis.py --old pi-lfm/data --new pi-lfm/data_v2.0.1 --out results/v2.0.1/step1.3/split_analysis.json`（horizon = T_in + T_out = 40）
- **所有 index JSON 和 params JSON 新舊版逐位元相同**，所以樣本數、sim_id 集合、time_id 分布全部不變。

| split | 樣本數 | sim 數 | time_id 範圍 | 最常見 stride |
|---|---|---|---|---|
| train_real | 9063 | 72（= remain_params_real） | 0–3950 | 20 |
| val_real | 4820 | 92 | 0–3910 | 20 / 40 |
| test_real | 4827 | 92 | 0–3910 | 20 / 40 |
| train_numerical | 18126 | 92 | 0–3950 | 20 |
| val/test_numerical | 408 / 408 | 43 | — | — |

params：remain_real 72、in_dist_test_real 10、out_dist_test_real 10；numerical 的 in/out 都是空的，remain 是 92。

**val 與 test 的關係（real，horizon 40）**
- 共用 sim_id：**92/92，全部共用**
- test 樣本中，時間窗和某個 val 窗有交集的：**3635/4827（75.3%）**
- 在共用的 sim 裡，test 用到的影格有 **94,560/146,110（64.7%）** 也出現在 val

**train 與 val/test 的關係（real）**：共用 72 個 sim。**536/4827（11.1%）個 test 樣本**、512/4820 個 val 樣本的時間窗和 train 窗有交集，也就是 remain 軌跡的時間分段之間有部分重疊。只做記錄，**沒有修改官方 split**。

**2.0.1 變更的軌跡 `3656.h5`**：屬於 in_dist_test，**不在 train 裡**。它佔 val 的 110/4820（2.28%）、test 的 86/4827（1.78%）。

對 Step 6.1 的影響：val 和 test 高度相關，所以 FM 的 val 子集只需要小而分層（見 D-011）。
