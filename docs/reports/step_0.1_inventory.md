# Step 0.1 報告：Repo 與 checkpoint 盤點

完整內容見對話紀錄（2026-09-29）。重點摘要：

- 官方：`62f4c80ab17f78933d046f2b038531dbc6a478a0`，外加本地 patch（`patches/official_62f4c80_fluid_hf_dataset_sim_id_column.patch`，見 D-003）
- 我們：`pi-lfm-code`，初始 commit `53f6def`
- 官方 checkpoint：`/work/b314513067/pi-lfm/ckpt/official/cylinder/{unet,deeponet,transolver}/{numerical,real,finetune}.pth`，沒有中間 checkpoint
- 重訓：`/work/b314513067/pi-lfm/runs/{model}/{exp}_{numerical|real}_{False|True}/<ts>/model_XXXX.pth`，每個 run 有 50 個中間 checkpoint，每個都存有 `best_iteration`

| 模型 | numerical best | real best | finetune best（起點） |
|---|---|---|---|
| U-Net 重訓 | 4600（`…numerical_False/2026-09-15_20-22-46`） | 6400（`…real_False/2026-09-15_20-59-14`） | 2000（`…real_True/2026-09-16_00-59-49`，起點是 num 4600） |
| DeepONet 重訓 | 300（`2026-09-15_20-21-44`） | 1500（`2026-09-15_20-22-41`） | 4700（`2026-09-16_01-01-00`，起點是 num 300） |
| Transolver 重訓 | 100（`2026-09-15_20-22-46`） | 600（`2026-09-15_20-59-15`） | 500（`2026-09-15_23-33-28`，起點是 num 100） |
| U-Net 官方 | 6400 | 6600 | 2400 |
| DeepONet 官方 | 200 | 3300 | 3600 |
| Transolver 官方 | 200 | 800 | 800 |

既有的評估結果（2.0.0、real test、N_ar=1）：`runs/{model}/{model}_cylinder_eval/<ts>/eval.log`，彙整在 `runs/RESULTS.md`。

異常：
1. 官方原始碼有既有的本地修改（D-003）。
2. 有兩個 eval.log 是兩次評估交錯寫進同一檔。
3. ~~val_losses 只有 14 筆~~ 已釐清：那是一個 dict，裡面有 14 種指標。
4. `pick_ckpt.py` 用字典序排序，Phase 4 前要修。
5. `train_cylinder.sh` 會共用並覆蓋 config，Phase 4 前要改。
