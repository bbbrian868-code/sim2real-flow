# 接續紀錄：DiT x-pred vs v-pred × width（2026-10-26 暫停）

暫停原因：使用者要先處理其他投稿的實驗。本檔記錄到暫停為止的狀態，下次從「接續步驟」開始。

- 任務清單原文：[TASK_2026-10-26_dit-xpred-width.md](TASK_2026-10-26_dit-xpred-width.md)（全域規則仍適用：每步停下回報、正式訓練與推論要先同意、新功能以預設關閉的開關加入）
- 決策紀錄：repo 根目錄 [DECISIONS.md](../../DECISIONS.md)，D1–D21。**DECISIONS.md 以後的條目優先於任務清單原文。**
- 背景文件 `dit_xpred_plan.md`：在 `/work/b314513067` 下找不到。

## 1. 進度

| Step | 狀態 |
|---|---|
| 0 盤點 | 完成，已回報 |
| 1 `pred` 開關與共用 v-space loss | **完成，已回報；停在 Step 1 的 STOP**，使用者還沒說「繼續」 |
| 2 寬度 config 與成本表 | 未開始 |
| 3–7 | 未開始 |

## 2. 目前定案的設定（R1–R8）

| 項目 | 值 | 依據 |
|---|---|---|
| 網格 | v-pred R1–R4、x-pred R5–R8；width 384 / 512 / 768 / 1024 | D2、D21（不加 1280 / 1536） |
| heads / head dim | 6 / 8 / 12 / 16，head dim 64 | D5 |
| patch / QK-norm | **4 / 開**（輸出 token 維度 60·4² = 960） | D19（取代 D11） |
| 寬度門檻 | 比較輸出 token 維度 960（不是輸入的 1920） | D12 |
| depth / MLP ratio / 條件注入 | 8 / 4.0 / y_t 與條件在 patch embedding 前 concat，t 走 adaLN-Zero | D7（Step 0 盤點） |
| loss | 兩種 pred 都用 target = (y1 − y_t) / max(1 − t, t_clip)，`t_clip` = 0.05 | D3、D4、D18 |
| 時間 | t=0 雜訊、t=1 資料（和任務範例相同，不需翻轉）；t ~ U(0,1) | Step 0 |
| paradigm / 訓練 | real，從頭訓練 50k 步，global batch 64，AdamW lr 1e-4，wd 0，betas (0.9, 0.999)，warmup 1000 + cosine，clip 1.0，EMA，bf16 autocast | D16、D6 |
| checkpoint | 每 2000 步以 EMA 在 536 筆 real val 子集（stride 9，Euler N=10，K=1，noise seed 0）算 RMSE，取最小（`best.pt`） | D13 |
| 評估 | N_ar=1：K=8 ensemble mean + spread；N_ar=10 rollout；能譜；推論時間 | D8、D9、D10 |
| 散度評估 | **本次不做**，延到下次 | D17（取代 D14） |
| baseline 的 N_ar=10 與 metric_batch_size | 延後決定（Step 6 前） | D15 |

R1–R8 的 config 預計要寫的欄位（Step 2 還沒建檔）：`backbone: dit`、`patch: 4`、`dim`、`heads`、`depth: 8`、`mlp_ratio: 4.0`、`qk_norm: true`、`pred: v|x`、`target_clip: true`、`t_clip: 0.05`；data / train / val 區段照 `fm/configs/cylinder/c2/dit_M_xp4_qk.yaml`（`train.iters` 在既有 config 是 100000 的 placeholder，實際用 `--iters 50000`）。

### 理由還沒補的條目

D12、D13、D15、D16、D17、D18、D20、D21 的「理由」欄都是「待補」。使用者說會自己補。D20、D21 另外記了 Claude 提案時的說明，標明不是使用者的理由。

## 3. 程式修改（pi-lfm-code，**尚未 commit**，HEAD = 75d4927）

| 檔案 | 改了什麼 |
|---|---|
| `DECISIONS.md`（新檔） | D1–D21 |
| `fm/paths.py` | `fm_loss(..., t_clip=None)`：None 時用原本的 target `y1 − y0`（原行不動）；給數值時 target = (y1 − y_t)/max(1−t, t_clip)，在 fp32 計算 |
| `fm/build.py` | 新增 `target_t_clip(mcfg)`；config 欄位 `target_clip`（預設 false）、`t_clip`（預設 0.05）；開啟時 x-pred 的分母也用 `t_clip`；只給 `t_clip`、或 `target_clip` 搭配 `t_floor` 都會報錯 |
| `fm/train_fm.py` | 把 `t_clip` 傳給 `fm_loss`；非 fm objective 開 `target_clip` 會報錯 |
| `fm/tests/test_fm.py` | 新增測試 (g)：target 一致性、oracle loss、oracle 取樣、config 檢查 |
| `docs/dit_xpred_width/`（新資料夾） | 任務原文與本檔 |

Step 1 驗證結果（細節見 Step 1 回報）：
- `target_clip` 關閉時，4 個既有 config × fp32 / bf16 的第一個 real batch：loss 和梯度 bit-exact。權重加擾動後 loss 仍 bit-exact，DiT 梯度的差異和「舊程式跑兩次」同量級，來自 GPU attention backward 的非決定性。
- t < 0.95 時 target 和 x − ε 的最大誤差 3.6e-6 至 5.7e-6；oracle loss 0；oracle 取樣在 Euler N=10 / 20 的 rel L2 ≤ 7e-8。N=32 時 oracle 也有 1.8e-2 的誤差（最後一步落在 clip 區間）。

驗證腳本和輸出（共用磁碟）：`/work/b314513067/pi-lfm/results/fm/2026-10-26_dit-xpred-width/`
- `S1_pred-switch-check/`：比對腳本、匯出的舊程式（75d4927）、第一個 batch、dev job 的輸出（job 410167、410493）、`oracle_real_batch.log`
- `S1b_data-sha256-check/`：D20 第 1 步的資料雜湊比對

## 4. 環境（Nano5）

- 叢集：Nano5，H100 partition `normal`（48 h）和 `dev`（2 h），H200 是 `normal2`；account `mst114566`。QOS 一次只能跑 2 個 job，8 個 run 要放在同一個 8 卡 job。GPU 預設用 H100（使用者偏好）。
- **pi-lfm 先前的 run 和成本數字都在 Nano4 的 H200 上**（D-025、D-037），不能直接和 H100 比。
- `slurm/fm.sh` 還是 Nano4 的設定（partition `8gpus`），要改才能在 Nano5 上用。
- 登入節點的記憶體 cgroup 約 12 GB：在 CPU 上做 DiT-M patch 2、batch 64 的 backward 會被 OOM 砍掉，要用 dev partition 的 GPU。
- Claude 的 scratchpad 在登入節點的 `/tmp`，**compute node 看不到**；job 用到的檔案要放在 `/work`。
- conda 環境：`/work/b314513067/conda/envs/realpdebench`；先 `source /work/b314513067/RealPDEBench/env.sh`。

### 資料（D20）

- FM config 寫死的 `/work/b314513067/pi-lfm/data_v2.0.1` 和 `/work/b314513067/pi-lfm/cache/v2.0.1` **都不存在**（搬到 Nano5 時沒有複製）。
- 現有資料：`/work/b314513067/clft_data`。D20 第 1 步已完成：183 個記錄的檔案中 182 個和 HF revision 0ae4426 的雜湊一致（約 180 GB），沒有不一致的檔案。缺的 `cylinder/channels.json` 沒有任何程式讀取，pi-lfm 當時的本地資料也沒有這個檔。`mean_std.pt` 用 HF cache 裡的那份，sha256 `2dff4af…` 和記錄相同。
- **還沒做**：D20 第 2 步（在 `pi-lfm/data_v2.0.1/` 建 symlink 資料根目錄，附 README）和第 3 步（`scripts/build_cache.py` 只重建 real cache 到 `pi-lfm/cache/v2.0.1/real`，約 23 GB，再跑單元測試 (f) 驗證）。預定在 Step 3 之前做（Step 2 不需要資料）。

## 5. 可比較的舊結果（real，50k 步，Nano4 H200）

| run | 設定 | 結果 |
|---|---|---|
| `pi-lfm/results/fm/c2/xp4_qk_s0–2` | x-pred、patch 4、QK-norm、512（和 R6 只差 D18 的 target） | test N20 K1 Rel L2 0.0782–0.0784，0/3 發散 |
| `pi-lfm/results/fm/step5.4_overfit*/dit_M*` | v-pred、patch 4、512，單一固定 batch | loss 卡在 0.50，沒有 test 指標 |
| `pi-lfm/results/fm/c2/qk_s0–2` | v-pred、patch 2、QK-norm、512 | Rel L2 0.0799–0.0807 |
| `pi-lfm/results/fm/phase6/dit_M_real_*`、`c2/base_s3–5` | v-pred、patch 2、無 QK-norm、512 | 9/9 在約 27k–30k 步發散 |

## 6. 待處理事項

1. **登入節點上的殘留程序**：`sfo_s2_stencils.py`（PID 288291、288607–288611，10-23 當掉後沒有結束）和 `run_D.py`（PID 4074648、4074796，10-22，有 zombie），0% CPU，占約 5.4 GB 記憶體。Claude 建議砍掉，**使用者還沒同意**。
2. Step 6 已知的問題（到時再決定）：
   - `scripts/eval_fm.py`、`eval_prob.py`、`eval_ckpts.py` 都只支援 N_ar=1（`pilfm/official_eval.py` 的 `run_test` 本身支援 N_ar）；baseline 從來沒有 N_ar=10 的結果；N_ar > 4 時官方的 metric_batch_size 等於各模型的 test_batch_size（D15）。
   - 官方程式碼沒有 `test_mask_prob` 這個參數（real test 的 p 全為 0，只評 u、v）。
   - 先前 FM 的 test 用 K=1 / K=5（另有 todo_D 的 K=16），本次是 K=8（D9）。
3. 程式修改尚未 commit。是否 commit、何時 commit 由使用者決定。

## 7. 接續步驟

1. 開新的 Claude session 時說：「讀 `pi-lfm-code/docs/dit_xpred_width/HANDOFF_2026-10-26_dit-xpred-width_after-step1.md` 和 `pi-lfm-code/DECISIONS.md`，接續 DiT x-pred × width 實驗」。
2. 先確認狀態沒有變：`cd /work/b314513067/pi-lfm-code && git status --short` 應該只有第 3 節列的檔案。
3. 使用者回覆「繼續」後做 **Step 2**：
   - 建 8 個 config（patch 4、QK-norm、target_clip，見第 2 節），只建檔、不訓練。
   - 成本表：參數量、GFLOPs/forward（`torch.utils.flop_counter.FlopCounterMode`）、單次 forward 時間（batch 1、bf16、H100，CUDA event，warm-up 20 次、量 100 次取中位數）。計時要用 dev partition 的 GPU job，檔案放在 `/work`。
   - 停下回報。
4. Step 3 之前先完成 D20 第 2、3 步（資料根目錄和 real cache），並把 `slurm/fm.sh` 改成 Nano5 的設定（先回報）。
