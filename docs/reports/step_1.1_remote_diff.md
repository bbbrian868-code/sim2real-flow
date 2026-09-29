# Step 1.1 報告：遠端變更比對

- HF commit（2026-09-18）：
  - `bd35c5abf159` "Update cylinder real shard 00018"
  - `e8b3bf528815` "Update controlled_cylinder real data shards"
  - `78d53f752c99` "Update foil real data shards"
  - `0ae4426085d2` "Bump data version to 2.0.1"
- 比對基準：舊版 = `5e6309f6e4de`（2026-08-14，是 2.0.0 的最後一個 commit，我們 09-15 的下載就來自這一版）；新版 = `0ae4426085d2`。
- 本地 `OLD_DATA_ROOT` 的 184 個檔案，sha256（LFS 檔）或 git blob sha1（小檔）和遠端舊版**全部一致**。唯一例外是 `cylinder/channels.json`，本地沒有這個檔（2026-05 新增，官方 downloader 不抓）。
- 比對表：`/work/b314513067/pi-lfm/results/v2.0.1/step1.1/sha_compare.csv`（185 列）；遠端樹：`remote_trees.json`；本地 hash：`local_old_sha256.txt`

**cylinder 範圍內有變更的檔案只有：**

| 路徑 | 大小（舊→新） |
|---|---|
| `cylinder/hf_dataset/real/data-00018-of-00073.arrow` | 784,794,712 → 784,795,768 |
| `version.json` | 495 → 495（data_version 2.0.0 → 2.0.1） |
| `README.md` | 15,335 → 15,422 |

- `hf_dataset/numerical/*`、所有 `*_index_*.json`、`*_params_*.json`、`mean_std.pt`、`channels.json`：**都沒有變更**。
- 資料集卡片寫的是「更新了 real shards」。以 cylinder 來說，實際上**只更新了 1 個 shard**（73 個裡的第 18 個）。
- 結論：numerical 資料和 split 都沒變，所以 **Phase 4 的 numerical 模型不需要重訓**（Step 4.1 的例外條件不成立）。
