# Step 0.2 報告：協議盤點

- 官方 code：`RealPDEBench@62f4c80a` + 本地 patch（`patches/official_62f4c80_fluid_hf_dataset_sim_id_column.patch`，只影響 sim_id 對照表的建立方式，不影響數值）
- 資料：2.0.0（`/work/b314513067/pi-lfm/data`）
- 行號以官方 repo `realpdebench/` 下的相對路徑為準；config 行號以 `pi-lfm-code/configs/cylinder/` 為準，這些檔案和官方 config 只差 `results_path`、`dataset_root` 兩行。

## 1. 形狀

| 項目 | 值 | 出處 |
|---|---|---|
| T_in | 20 | `data/fluid_hf_dataset.py:365` (`in_step=20`)，config 沒有覆蓋 |
| T_out | 20 × N_ar | `fluid_hf_dataset.py:366, 109` |
| loader 輸出 | `(T, H, W, C) = (20, 64, 128, 3)`，input 與 target 都是這個形狀 | `fluid_hf_dataset.py:305-309, 338`；實際讀檔確認 |
| 空間解析度 | real 原生 64×128（`sub_s_real=1`）；numerical 原生 128×256，經 `sub_s_numerical=2` 降到 64×128 | `fluid_hf_dataset.py:376-377, 119, 289-290` |
| 樣本數（2.0.0） | real train 9063 / val 4820 / test 4827；numerical train 18126 | 實際讀檔 |

**注意**：任務規格 5.2 假設 128×256（U-Net 瓶頸 8×16、DiT patch 8 → 512 token），實際是 **64×128**。處理方式見 DECISIONS.md D-005。

## 2. 通道

- 通道固定為 `(u, v, p)`，input 和 target 都一樣（`fluid_hf_dataset.py:305`）。**target 包含 p 通道。**
- **real**：p 一律填 0，input 和 target 都是（`fluid_hf_dataset.py:293-295`）。
- **numerical**：以 `mask_prob` 的機率把 p 整個設為 0。遮蔽以樣本為單位，input 和 target 同時遮（`fluid_hf_dataset.py:297-302`）。
- 訓練 loss 是全部 3 個通道的 MSE，包含 p（`model/unet.py:569-571` 等）。real 的 p target 在正規化後是常數 `(0-μ_p)/σ_p`，模型會學著輸出這個常數。
- 評估時，如果第一個 batch 的 target 某通道全為 0，就把它算成「未量測」。real test 因此 `c=2`，指標只算 u、v（`eval.py:298-303`、`train.py:352-357`、`utils/metrics.py:34`）。

## 3. sim 加噪與模態遮蔽

| 項目 | 值 | 出處 |
|---|---|---|
| 加噪對象 | **只有 numerical 資料**，**input 和 target 都加** | `fluid_hf_dataset.py:312-315` |
| 加噪形式 | gaussian、**乘法型**：`x ← x + x·ε·noise_scale`，ε~N(0,1)，逐元素獨立 | 同上 |
| noise_scale | 0.1（三個模型都是） | `unet.yaml:18`、`deeponet.yaml:14`、`trainsolver.yaml:14` |
| noise_type | gaussian（config 未設定，用預設值） | `fluid_hf_dataset.py:378` |
| mask_prob | U-Net **0.5**，DeepONet **0.1**，Transolver **0.1** | `unet.yaml:17`、`deeponet.yaml:13`、`trainsolver.yaml:13` |
| 遮蔽作用 | p 通道在 input 和 target **同時**設為 0；只在 numerical 資料上做 | `fluid_hf_dataset.py:297-299` |
| val/test | 沒有加噪也沒有遮蔽：val/test 只用 real，而 `noise_scale` 預設 0 | `train.py:178-184`、`eval.py:164-171` |

**Phase 5 的結論：官方噪聲有加在 target 上。** 因此 FM 報告要標「目標含 sim 加噪」（只限 numerical 和 finetune 的 sim 預訓練階段），而且 Step 6.3 第 4 項（只對條件加噪）必須做。

## 4. 正規化

- `GaussianNormalizer`，每個通道各自的 mean/std，input 和 target 各有一組（`data/data_normalizer.py:20-62`）。
- 統計量**永遠取自 numerical train**，不論訓練設定是哪一種（`train.py:185-191`、`eval.py:180-186`）。
- 從 `{dataset_root}/cylinder/mean_std.pt` 讀取；**檔案不存在時會靜默重算並寫回**（`data_normalizer.py:26-34`）。2.0.0 用的檔案是從 HF 下載的（見 `runs/RESULTS.md` §6）。
- 值（2.0.0）：mean_in = (0.22169, 1.6e-5, −0.02622)，std_in = (0.12930, 0.09847, 0.20826)；target 的值幾乎相同。
- 對遷移的影響：2.0.1 的 `mean_std.pt` 必須和 2.0.0 逐位元比對（Step 1.1）。如果 `NEW_DATA_ROOT` 缺這個檔，會被靜默重算，所以要在 Step 1.2 驗證檔案存在。

## 5. 最佳化

| | U-Net | DeepONet | Transolver | 出處 |
|---|---|---|---|---|
| optimizer | Adam | Adam | Adam | `train.py:290`（寫死；沒有 weight decay） |
| lr | 1e-4 | 1e-4 | 7e-4 | config |
| scheduler | cosine, T_max=num_update | 同左 | 同左 | `train.py:293-294` |
| batch（train/test） | 12 / 12 | 32 / 64 | 16 / 16 | config |
| num_update | 10000 | 5000 | 5000 | config |
| clip_grad_norm | 0（關閉） | 0 | 0 | config、`train.py:330` |
| seed | 0 | 0 | 0 | config；`utils/utils.py:26-31`（沒有 seed Python `random`；mask 用的 `random.random()` 在 DataLoader worker 內由 torch 導出 seed） |

Transolver config 的 `test_interval: 200`（`trainsolver.yaml:39`）**沒有被 `train.py` 使用**。

## 6. Checkpoint 選擇準則

- 每 `num_update/50` 步驗證一次（U-Net 每 200 步，其他每 100 步）（`train.py:344`）。
- 驗證集是 **real val**，完整 4820 筆（`train.py:178-184`）。
- 指標是**反正規化後的 val RMSE**（只算 u、v），**越小越好**（`train.py:371-373, 390-392`）。
- 每次驗證都存 checkpoint，並在裡面記錄 `best_iteration` 和 `best_val_loss`（`train.py:410-418`）。
- `val_losses` 是一個 dict，裡面有 14 種指標，每種都是 50 筆的 list（這解釋了 Step 0.1 看到的「14」）。

## 7. Finetune

- 旗標 `--is_finetune`：只載入 `model_state_dict`（`train.py:299-300`、`model/model.py:14-16`）。
- **optimizer 和 scheduler 重新初始化**：`train.py:290-296` 在載入權重之前就建立好了，沒有載入 optimizer 狀態。
- **learning rate、步數、batch 都和從頭訓練相同**：config 沒有另外的 finetune 參數。
- 訓練資料是 real（`--train_data_type real`），加噪和遮蔽都不作用（因為只對 numerical 作用）。

## 8. 評估參數

| 項目 | 值 | 出處 |
|---|---|---|
| test split | real test，寫死 | `eval.py:164-171` |
| N_ar | 1（`eval_*_nar1.yaml:34`）；訓練 config 裡的 N_autoregressive=5/10/3 只影響 eval，不影響 train | `eval_unet_nar1.yaml:34` |
| test_mode | `"all"`（預設值；eval.py 沒有傳入） | `fluid_hf_dataset.py:363`、`eval.py:164-171` |
| test_mask_prob | **官方程式碼沒有這個參數。** real test 的 p 固定為 0；mask_prob 只傳給 train_dataset（`eval.py:172-179`），而且這裡的 train_dataset 只拿來推斷模型形狀 | — |
| metric_batch_size | N_ar ≤ 4 時是**整個測試集**（4827）一次算；否則等於 test_batch_size | `eval.py:346-352` |
| normalizer 來源 | 永遠是 numerical train；`--train_data_type` 只影響模型建構時用來推斷形狀的 train_dataset | `eval.py:172-186` |
| probe_diagnostic | True（只記 log，不列入主表） | config |

## 9. 各模型不一致的地方

只有 **mask_prob**（U-Net 0.5，其他 0.1）、lr、batch、num_update 不同；其餘協議都一樣。
