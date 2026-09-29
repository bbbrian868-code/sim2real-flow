# Step 2.1 報告：完整性檢查（real）

- 指令：`python scripts/traj_stats.py --root {pi-lfm/data, pi-lfm/data_v2.0.1} --type real --out results/v2.0.1/step2.1/real_stats_{v2.0.0,v2.0.1}.csv`
- 產出：上述兩個 CSV（92 條軌跡 × u/v/vo 的 nan/inf/mean/std/min/max）；變更的軌跡並列在 `changed_traj_side_by_side.csv`
- 所有軌跡的形狀都是 `(3990, 64, 128)`，新舊版都一樣。NaN/Inf：新舊版都是 0。
- **只有 `3656.h5` 的統計量有變**，其餘 91 條逐值相同。

| 3656.h5 | 舊 | 新 | 其他 91 條的範圍（舊版） |
|---|---|---|---|
| u mean / std | 1.168 / 0.294 | 0.156 / 0.020 | mean 0.116–0.45 |
| v mean / std | 0.0113 / 0.597 | 0.0026 / 0.041 | std 0.017–0.2 |
| vo std | 0.0097 | 3.86 | — |

舊版的 3656 在 u_mean、u_std、v_std 三項都是全資料集的最大值，是明顯的離群值；新版回到正常範圍。
numerical 部分沒有另外計算：新舊版逐位元相同（見 Step 1.1），統計量必然一樣（D-007）。
