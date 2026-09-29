# Step 1.2 報告：下載 2.0.1

- 指令：`HF_HUB_DISABLE_XET=1 python scripts/fetch_v201.py --trees results/v2.0.1/step1.1/remote_trees.json --old pi-lfm/data --new pi-lfm/data_v2.0.1 --old-sha-list results/v2.0.1/step1.1/local_old_sha256.txt`
- 固定的 HF revision：`0ae4426085d250787f96d5456bc0d30ce5f277d2`（直連 huggingface.co，沒有用 mirror）
- 181 個檔案（167.4 GiB）：sha256 或 blob sha1 和遠端新版相同，**建 symlink 指向 `pi-lfm/data` 的舊檔**。
- 下載 4 個檔案（748.5 MiB）：`README.md`、`version.json`、`cylinder/channels.json`（舊版本地原本沒有）、`cylinder/hf_dataset/real/data-00018-of-00073.arrow`
- 驗證：全部 185 個檔案（經由 symlink 讀取）都和遠端新版的 hash 相符 → `verify: OK`。`version.json` 內容是 `data_version=2.0.1`、`min_code_version=0.2.0`，和官方 code 0.2.0 相容。
- 產出：`/work/b314513067/pi-lfm/data_v2.0.1/`（包含 `fetch_manifest.json`）、`results/v2.0.1/step1.2_fetch.log`
- 注意：`data_v2.0.1` 的大部分檔案是指向舊資料的 symlink，**舊資料不能刪也不能移**，否則新資料根目錄會失效。
