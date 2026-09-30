# Step 5.5 報告：Smoke test、throughput、DDP 檢查

- 設定：numerical train（有 sim 加噪和遮蔽），bf16，global batch 64，**單張 H200**（本叢集沒有 L40S，見下），300 步，不做 val。
- 指令：`slurm/fm.sh --config fm/configs/cylinder/<cfg>.yaml --train-data-type numerical --no-val --iters 300 --log-every 20`
- 產出：`results/fm/step5.5_smoke/`（沒有快取，資料讀取是瓶頸）、`results/fm/step5.5_smoke_cache/`（U-Net，有快取）、`results/fm/step5.5_smoke_cache_p2/`（DiT patch 2，有快取）

## 資料讀取瓶頸（已修正）
第一輪 smoke（官方 Arrow 路徑的 zero-copy 切片）6 個設定都只有 **約 0.25 it/s**（每秒 16 筆樣本），瓶頸在 Lustre 的隨機讀取：numerical 每筆樣本要讀約 16 MB 的原生 128×256 幀，而且當時約有 20 個 job 同時讀取。
修正方式：建立衍生快取 `scripts/build_cache.py`（降採樣後的 64×128、channel-last `.npy`；numerical 34 GB、real 23 GB，放在 `/work/b314513067/pi-lfm/cache/v2.0.1`）。**單元測試 (f) 驗證它和官方 dataset 的輸出逐位元相同**，包括遮蔽和加噪的亂數抽取。

## Throughput（有快取，單張 H200，bf16，batch 64）

| 設定 | 參數量 | it/s | samples/s | peak memory |
|---|---|---|---|---|
| U-Net-S | 9.74M | 9.08 | 581 | 8.1 GB |
| U-Net-M | 38.80M | 9.62 | 616 | 15.1 GB |
| U-Net-L | 125.45M | 5.06 | 324 | 27.3 GB |
| DiT-S（p2） | 9.92M | 10.55 | 675 | 14.1 GB |
| DiT-M（p2） | 39.10M | 5.51 | 353 | 27.0 GB |
| DiT-L（p2） | 119.43M | 2.61* | 167 | 54.6 GB |

\* DiT-L 是讀取時只跑到 140 步的部分結果。U-Net-S 和 U-Net-M 的 throughput 相近，因為兩者都被 12 個 DataLoader worker 的讀取速度（約 600 samples/s）限制住，GPU 並沒有滿載。

## 完整訓練時間推估（以 50k 步為準，見 D-023；不含 val，val 每 2000 步一次，536 筆 × 10 步 Euler，每次 < 1 分鐘）

| 設定 | H200 ×1 | H200 ×4 | H200 ×8 | L40S ×2（推估） |
|---|---|---|---|---|
| U-Net-S | 1.5 h | 0.45 h | 0.25 h | 1.6 h |
| U-Net-M | 1.4 h | 0.45 h | 0.25 h | 2.1 h |
| U-Net-L | 2.7 h | 0.8 h | 0.4 h | 4.1 h |
| DiT-S | 1.3 h | 0.4 h | 0.2 h | 2.0 h |
| DiT-M | 2.5 h | 0.7 h | 0.4 h | 3.8 h |
| DiT-L | 5.3 h | 1.5 h | 0.8 h | 8.0 h |

推估方式：H200 多卡以 DDP 90% 效率換算；資料讀取受限的設定假設每張卡各自有 12 個 worker，所以讀取量也跟著卡數成長。**L40S 沒有實測**：本叢集（Nano4）只有 H200。L40S ×2 用 bf16 dense 算力比例（362 / 989 TFLOPS ≈ 0.37）乘上 2 卡 × 90% 換算，誤差可能到 ±50%。L40S 只有 48 GB，DiT-L 單卡 batch 64 需要 55 GB，必須用 2 卡各 32 筆。

## DDP 一致性檢查
- 設定：U-Net-S，numerical，`--ddp-check`（每個 rank 讀同一個 global batch 64 和同一組噪聲，各自取自己的 1/world 份），seed 0，20 步，每步記錄 all-reduce 平均後的 loss。
- 1 張 GPU vs 2 張 GPU（DDP）：**20 步的 loss 全部一致，最大相對差異 1.2e-7**（其中 13 步完全相同）→ 通過。
- 產出：`results/fm/step5.5_ddp/{g1,g2}/log.jsonl`

## 執行異常
- 有些節點會讓 `srun` 行程拿不到可用的 GPU（`no accelerator` 或 `CUDA unknown error`），發生在 hgpn003、hgpn146、hgpn111，都是多個 job 同時落在同一個節點的時候。處理方式：`slurm/gpu_guard.sh` 在 job 一開始先試著初始化 CUDA，失敗就排除這個節點、原樣重新提交，然後退出。
