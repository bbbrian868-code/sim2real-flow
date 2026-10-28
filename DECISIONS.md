# 決策紀錄：DiT 預測目標（x vs v）× 模型寬度

任務：`task_dit_xpred_width.md`。背景與文獻依據見 `dit_xpred_plan.md`。
本檔只記錄這個實驗的決策；本 repo 先前各階段的決策在 `docs/DECISIONS.md`（D-001 起，編號格式不同）。

規則：每次在 STOP 點的決定都新增一筆；理由必須是使用者給的，沒有就填「理由：待補」；已記錄的條目不得刪改，
推翻時新增一筆，並在「取代」欄註明被取代的編號。
補充（2026-10-27，D23）：使用者授權 Claude 依討論內容填寫理由，包括使用者詢問 Claude 建議時 Claude 提出的理由；
純屬使用者偏好的決定，理由欄留空，標「（使用者偏好）」。依此補填的理由都註明「〔D23 補填〕」與出處。

### 條目格式

```
### D{編號}　{一句話標題}
- 日期：YYYY-MM-DD
- 階段：{規劃 / Step N}
- 決定：
- 理由：
- 被否決的替代方案：
- 取代：{無 / D?}
- 影響範圍：{哪些 run、哪些檔案或指標}
```

---

### D1　比較 x-pred 與 v-pred 兩種輸出參數化
- 日期：2026-10-26
- 階段：規劃
- 決定：在相同架構下比較網路直接輸出 $x$ 與直接輸出 $v$。
- 理由：先前 DiT 的 hidden 只有 512，但 token 維度為 960；預測速度場等於要同時重建雜訊，而雜訊沒有結構，需要較大容量。JiT（Li & He, CVPR 2026）顯示 token 維度 ≥ width 時 $v$/$\epsilon$-pred 會災難性失效，而 $x$-pred 不會。
- 被否決的替代方案：只換成 $x$-pred 而不做對照（無法證明改善來自參數化本身）。
- 取代：無
- 影響範圍：R1–R8。

### D2　寬度掃描 384 / 512 / 768 / 1024
- 日期：2026-10-26
- 階段：規劃
- 決定：同時掃描寬度，形成 2×4 網格。
- 理由：RAE（Zheng et al., ICLR 2026）主張 $v$-pred 下 width 必須 ≥ token 維度才能收斂，與 JiT「不需加寬」的主張形成對立預測；1024 刻意選在 960 之上以測試該門檻。8 個 run 剛好對應 8 張 H100。
- 被否決的替代方案：只比較 512 下的兩種參數化（無法回答「加參數量是否同樣有效」）。
- 取代：無
- 影響範圍：R1–R8。

### D3　兩種參數化共用同一個 v-space loss
- 日期：2026-10-26
- 階段：規劃
- 決定：$x$-pred 的輸出先換算成 $v$ 再算 loss，與 $v$-pred 使用同一個 loss。
- 理由：讓兩者只差在「網路輸出什麼」，排除 loss 權重不同造成的差異；這也是 JiT 的作法，且他們發現 $x$-pred 搭 $v$-loss 效果最佳。
- 被否決的替代方案：$x$-pred 搭 $x$-loss（會同時改變參數化與 loss 權重，兩個變因混在一起）。
- 取代：無
- 影響範圍：Step 1 實作、R5–R8。

### D4　分母 clip 值 t_clip = 0.05
- 日期：2026-10-26
- 階段：規劃
- 決定：計算 $1/(1-t)$ 時分母下限為 0.05，prediction 與 target 使用同一個 clip 後的分母。
- 理由：$x$-pred 搭 $v$-loss 等價於 $x$-loss 乘上 $1/(1-t)^2$，$t \to 1$ 時會爆掉；0.05 沿用 JiT 設定。prediction 與 target 用同一分母可避免 $t > 0.95$ 區間的不一致。
- 被否決的替代方案：不 clip；或只 clip prediction 端。
- 取代：無
- 影響範圍：Step 1 實作、R5–R8。

### D5　固定 head dim = 64，heads 數隨寬度變動
- 日期：2026-10-26
- 階段：規劃
- 決定：heads = 6 / 8 / 12 / 16。
- 理由：維持每個 head 的表示能力一致，使寬度成為唯一的容量變因；這也是 ViT/DiT 系列的標準縮放方式。
- 被否決的替代方案：固定 heads 數（head dim 會隨寬度改變，多一個混淆變因）。
- 取代：無
- 影響範圍：Step 2、R1–R8。

### D6　LR 沿用原設定，不隨寬度調整
- 日期：2026-10-26
- 階段：規劃
- 決定：所有寬度使用相同 LR 與 schedule。
- 理由：控制變因數量並維持與既有設定可比；計算預算以主網格為優先。
- 已知限制：較寬模型的最佳 LR 可能不同，結果解讀時需註明。
- 被否決的替代方案：每個寬度各自掃 LR（成本約為主實驗數倍）。
- 取代：無
- 影響範圍：R1–R8。

### D7　其餘架構與訓練設定全部固定
- 日期：2026-10-26
- 階段：規劃
- 決定：depth、patch size、條件注入方式、MLP ratio、時間取樣分布、seed、資料順序、總 iters 與原設定相同。
- 理由：一次只改 `pred` 與 `width` 兩個因子，結果才可歸因。
- 被否決的替代方案：同時套用 JiT 的其他改良（RoPE、SwiGLU、noise shift 等）。
- 取代：無
- 影響範圍：R1–R8。

### D8　評估協定與既有 baselines 完全相同，且不用 test split 選 checkpoint
- 日期：2026-10-26
- 階段：規劃
- 決定：沿用 U-Net / DeepONet / Transolver 的 eval 腳本與參數（`N_ar`、`test_mask_prob`、`metric_batch_size`、同一 test set）；checkpoint 選擇不得用到 test。
- 理由：先前重現 baseline 時發現評估協定對指標的影響大於架構差異；且 val 與 test 高度相關（同一批模擬、交錯切窗），用來選 checkpoint 等同洩漏。
- 被否決的替代方案：為 FM 另寫一套評估流程。
- 取代：無
- 影響範圍：Step 5、Step 6。

### D9　生成式評估以 K = 8 樣本的 ensemble mean 計算指標，並報 spread
- 日期：2026-10-26
- 階段：規劃
- 決定：每個測試樣本生成 8 個樣本。
- 理由：FM 是隨機生成，單一樣本的指標變異大；ensemble mean 才能與確定性 baselines 公平比較，spread 則保留 FM 能估計不確定性的優勢。
- 被否決的替代方案：只取單一樣本。
- 取代：無
- 影響範圍：Step 6。

### D10　不以匹配參數量作為成本限制，改為完整報告推論成本
- 日期：2026-10-26
- 階段：規劃
- 決定：報告參數量、GFLOPs/forward、NFE、每幀時間、整條 rollout 時間，並畫準確度 vs 推論時間圖。
- 理由：本方法的省時對照組是 CFD 求解器，ML 方法之間毫秒級的差距可以接受；先前為了回答 cost 問題而把 FM 骨架壓到與 U-Net 同量級，反而可能是 DiT 落入容量瓶頸的原因。同時，FM 的實際推論成本是單次 forward × NFE × ensemble 數 × rollout 步數，只比單次 FLOPs 會低估，應主動完整報告。
- 被否決的替代方案：維持與 U-Net 參數量匹配、只比較 FLOPs。
- 取代：無
- 影響範圍：Step 2、Step 6。

### D11　8 個 run 共用 patch 2 + QK-norm（Step 0 選項 B）；patch 4 不重跑，用上次的結果比較
- 日期：2026-10-26
- 階段：Step 0
- 決定：R1–R8 一律 patch 2、開啟 QK-norm。使用者原話：「選B，用上次的結果來比較就好」。「上次的結果」具體指哪些 run 尚待使用者確認（見 Step 0 補充回報）。
- 理由：（使用者偏好）〔D23 補填；已被 D19 取代〕
- 被否決的替代方案：(A) patch 4 + QK-norm；(C) patch 2 與 patch 4 都跑（16 個 run）。
- 取代：無
- 影響範圍：Step 2 的 config、R1–R8。

### D12　寬度門檻以輸出 token 維度 960 為準
- 日期：2026-10-26
- 階段：Step 0
- 決定：D2 所說的「token 維度」取輸出端（y_t 部分）的 60·p² = 960（patch 4），不是輸入端的 120·p² = 1920。
- 理由：先前討論的「960」就是輸出端的 token 維度（D-019）：v-pred 要重建的是 y_t 部分的速度場（含雜訊），條件只出現在輸入端。Step 5.4 的觀察也對應輸出端：v-pred、patch 4、width 512 的 loss 卡在 0.50，接近以輸出維度推算的下限 1 − 512/960 ≈ 0.47。〔D23 補填；出處：Step 0 回報「960 是哪一個維度」與先前 DiT 的 log〕
- 被否決的替代方案：以輸入 token 維度 1920 為準（需要再加 width 2048）。
- 取代：無
- 影響範圍：D2 的解讀、Step 6 的結果解讀。

### D13　checkpoint 沿用現行規則：real val 子集上的最佳點
- 日期：2026-10-26
- 階段：Step 0
- 決定：每 2000 步以 EMA 權重在 536 筆 real val 子集（stride 9，Euler N=10、K=1、noise seed 0）上算 RMSE，取最小者（`best.pt`）；不用 test。
- 理由：（使用者偏好）〔D23 補填；Step 0 只列出選項，Claude 未提出建議〕
- 被否決的替代方案：(b) 固定使用 `last.pt`；(c) 其他規則（例如 D-031 的模擬訓練 loss 選點）。
- 取代：無
- 影響範圍：Step 5、Step 6、R1–R8。

### D14　散度評估在 real 與 numerical 上都做
- 日期：2026-10-26
- 階段：Step 0
- 決定：Step 6 第 3 項的散度殘差同時在 real（real 格距、real floor）與 numerical（sim 格距、sim floor）上計算。
- 理由：（使用者偏好）〔D23 補填；已被 D17 取代〕
- 被否決的替代方案：只做其中一種。
- 取代：無
- 影響範圍：Step 6 第 3 項。

### D15　N_ar = 10 的 baseline 評估與 metric_batch_size 問題延後決定
- 日期：2026-10-26
- 階段：Step 0
- 決定：baseline 是否補跑 N_ar = 10，以及 N_ar > 4 時各模型 metric_batch_size 不同的處理方式，延到之後決定（Step 6 前）。
- 理由：這兩項只影響 Step 6，Step 1–5 用不到，所以 Claude 在 Step 0 提出可以等到 Step 6 再決定，使用者同意。背景是：三個 eval 腳本都只支援 N_ar = 1，baseline 從未跑過 N_ar = 10，而 N_ar > 4 時官方的 metric_batch_size 等於各模型的 test_batch_size，無法做到「完全相同」。〔D23 補填；出處：Step 0 回報疑點 7 與 Q8〕
- 被否決的替代方案：無（尚未決定）。
- 取代：無
- 影響範圍：Step 6 第 2 項、圖 2。

### D16　訓練 paradigm 用 real，總 iters 照 real 的原設定（50k 步）
- 日期：2026-10-26
- 階段：Step 0
- 決定：R1–R8 都用 `--train-data-type real` 從頭訓練，50k 步 × global batch 64，warmup 1000 + cosine（D-023），lr 1e-4 不縮放。
- 理由：使用者原本認為先前的「960 vs 512」討論是在 numerical 上，請 Claude 查證。查證結果是相關證據全部來自 real：Step 5.4 的 patch 4 平台用的是 real train 的 batch，C2 的 x-pred patch 4 是 real，v-pred patch 2 的 9/9 發散也只發生在 real；numerical 上的 patch 4 只有 300 步的 smoke。使用者據此改選 real。這樣主網格和要比較的舊 patch 4 結果（C2 `xp4_qk_s0–2`、Step 5.4）是同一個 paradigm，可以直接並列。50k 步是 real 的原設定（D-023，各 run 的 `run_meta.json`）。〔D23 補填；出處：Step 0 補充回報 Q4/Q5〕
- 被否決的替代方案：numerical（50k 步）；finetune（從 numerical 最佳點初始化，20k 步，D-024）。
- 取代：無
- 影響範圍：Step 4 的時間推估、Step 5、R1–R8。

### D17　本次實驗不做散度評估，延到下次
- 日期：2026-10-26
- 階段：Step 0
- 決定：Step 6 第 3 項（散度殘差，含 real 與 numerical 兩種）本次不做，延到下一次實驗。本次的 Step 6 只做第 1、2、4、5 項。
- 理由：（使用者偏好；使用者原話：「這次先不引入divergence的問題，那是下次的」）〔D23 補填〕
- 被否決的替代方案：照 D14 在 real 與 numerical 上都做。
- 取代：D14
- 影響範圍：Step 6 第 3 項、`results/dit_xpred_width.csv` 不含散度欄位。

### D18　target 也用 clip 後的分母（方案 a1），新增 config 欄位 `t_clip`
- 日期：2026-10-26
- 階段：Step 0
- 決定：以新的 config 開關加入「target 使用同一個 clip 後的分母」，預設關閉（舊行為不變，含舊 config 的 `t_floor`）；R1–R8 的 v-pred 與 x-pred 都打開，target = (y1 − y_t) / max(1 − t, t_clip)。分母下限的欄位名稱用 `t_clip`（預設 0.05）。
- 理由：使用者詢問 a1 和 a2 哪個好，Claude 建議 a1，使用者採用。(1) 用 a1 時，v-pred 和 x-pred 的 loss 對每個 t 都等於 ‖x̂ − y1‖² / max(1−t, 0.05)²，兩組 run 唯一的差別就是網路輸出 v 還是 x，也就是 D1 要測的東西。(2) a2 在 t > 0.95 的約 5% 樣本上讓兩組的 target 不同，和 D3、D4 矛盾，而且這個差異會和參數化的效果混在一起。(3) a2 唯一的好處是新的 v-pred 和舊 run bit-exact，但舊的 x-pred（C2 `xp4_qk`）在 a1 和 a2 下都無法重現，所以和舊結果比都只是近似比較。用開關加入、預設關閉，是為了遵守全域規則 3（方案 b 會讓 C2/C3 的舊 config 無法重現）。欄位名稱 `t_clip` 是使用者的選擇（和任務 Step 1 的欄位名稱一致）。〔D23 補填；出處：Step 0 補充回報 Q3 與 Claude 的 a1 建議〕
- 被否決的替代方案：(a2) 只有 x-pred 用 clip 後的 target、v-pred 維持 y1 − y0；(b) 直接修改現有 x-pred 的 target；沿用 `t_floor` 作為欄位名稱。
- 取代：無
- 影響範圍：Step 1 實作（`fm/paths.py`、`fm/build.py`）、R1–R8。

### D19　8 個 run 全部用 patch 4 + QK-norm，不跑 patch 2
- 日期：2026-10-26
- 階段：Step 0
- 決定：R1–R8 一律 patch 4（輸出 token 維度 60·4² = 960）、開啟 QK-norm；網格維持 D2 的 2×4（v-pred R1–R4、x-pred R5–R8，width 384 / 512 / 768 / 1024）。QK-norm 沿用 D11 / Step 0 選項 A 的設定。
- 理由（使用者）：實驗目的是比較兩種解法：「把模型加寬，繞開 960 維的限制」與「改用 x-pred」。patch 2 的 token 只有 240 維，不受這個限制，沒有比較意義。v-pred 是原本的參數化；x-pred 是為了解決 width 512 難以學習 960 維雜訊特徵才引入的。
- 被否決的替代方案：patch 2 + QK-norm（D11）；patch 2 與 patch 4 都跑（16 個 run）。
- 取代：D11
- 影響範圍：Step 2 的 config 與成本表、R1–R8。

### D20　資料來源：clft_data 先核對 sha256，再以 symlink 重建 `pi-lfm/data_v2.0.1/`，只重建 real cache
- 日期：2026-10-26
- 階段：Step 1
- 決定：(1) 把 `/work/b314513067/clft_data/cylinder` 的每個檔案和 pi-lfm 記錄的 HF revision 0ae4426 雜湊（`pi-lfm/results/v2.0.1/step1.1/sha_compare.csv`）比對，唯讀；(2) 在原路徑 `/work/b314513067/pi-lfm/data_v2.0.1/` 建 symlink 資料根目錄，指向 clft_data 的檔案，加上 HF cache 中 sha256 已核對相同的 `mean_std.pt`（2dff4af…），附 README 記錄來源；(3) 只重建 real 的 cache（`scripts/build_cache.py` → `/work/b314513067/pi-lfm/cache/v2.0.1/real`），並以單元測試 (f) 驗證和官方 loader 逐位元相同。不寫入 clft_data。
- 理由：使用者詢問怎麼做最好，Claude 建議此方案，使用者同意。(1) 雜湊核對可以確認 clft_data 和先前結果（C2 `xp4_qk` 等）訓練用的是同一份資料，和舊結果的比較才乾淨；而且只讀不寫。(2) 在原路徑建 symlink 資料根目錄，新舊 config 都不必改路徑，也不寫入 CLFT 共用的資料。(3) 本次只訓練 real（D16），numerical 只用來提供 normalizer 統計，而這些統計來自 `mean_std.pt`，所以只重建 real cache。被否決的 (b) 會寫入 CLFT 使用的資料；(c) 要重新下載約 170 GB，下載後仍需核對。〔D23 補填；出處：Step 1 回報後 Claude 的建議。原本另列的「Claude 建議時的說明」已併入本欄〕
- 被否決的替代方案：(b) 把 `mean_std.pt` 複製進 clft_data；(c) 重新下載。
- 取代：無
- 影響範圍：所有 FM config 的 `data.root` / `data.cache_dir`（路徑不變）、Step 3–6。

### D21　網格維持 2×4，最大 width 1024，不加 1280 / 1536
- 日期：2026-10-26
- 階段：Step 1
- 決定：v-pred 與 x-pred 的 width 都是 384 / 512 / 768 / 1024；不加更寬的 v-pred。若 Step 6 後 R4 的結果無法判讀，再另外提出，需使用者同意。
- 理由：使用者說「有1024就好」，並詢問 1280、1536 是否必要；Claude 認為現在不需要，使用者同意。(1) RAE 的主張是 width ≥ token 維度，1024 已在 960 之上，R4 可以直接回答；1280、1536 只能測「門檻之上還需要多少餘裕」，不是本次要問的問題。(2) 更寬的 run 需要超過 8 張 GPU。QOS 限制一次 2 個 job，所以要多開一個 job，或每張卡跑 2 個 run。(3) 依 D6，LR 固定不變，寬度越大，LR 的混淆越多。〔D23 補填；出處：Step 1 回報後 Claude 的建議。原本另列的「Claude 建議時的說明」已併入本欄〕
- 被否決的替代方案：加 1280 / 1536 的 v-pred；x-pred 只跑 512、把 GPU 讓給更寬的 v-pred。
- 取代：無
- 影響範圍：Step 2、R1–R8。

### D22　核准進入 Step 2
- 日期：2026-10-27
- 階段：Step 1 的 STOP
- 決定：使用者回覆「繼續」，進入 Step 2（建 R1–R8 config、成本表；只建檔與量測，不訓練）。
- 理由：（使用者偏好）〔D23 補填〕
- 被否決的替代方案：無（使用者未提出）
- 取代：無
- 影響範圍：Step 2。

### D23　理由欄改由 Claude 依討論補填；屬使用者偏好的決定留空
- 日期：2026-10-27
- 階段：Step 2 的 STOP
- 決定：所有「理由：待補」的條目，由 Claude 依當時的討論內容補填，包括使用者詢問建議時 Claude 提出的理由；純屬使用者偏好的決定，理由欄留空，標「（使用者偏好）」。補填的地方都註明「〔D23 補填〕」與出處。本次補填 D11–D18、D20–D22。D20、D21 原本另列的「Claude 建議時的說明」併入理由欄。
- 理由：（使用者偏好）
- 被否決的替代方案：維持原規則，由使用者自行補填。
- 取代：無（修改的是全域規則 8 中「理由必須是使用者給的」這一點，以及 D11–D22 的理由欄；其他欄位不動）
- 影響範圍：DECISIONS.md 全檔。

### D24　Step 2 成本表在 batch 1 之外加量 batch 64 的 throughput（選項 b）
- 日期：2026-10-27
- 階段：Step 2 的 STOP
- 決定：保留任務規定的 batch 1 時間（bf16、warm-up 20 次、100 次取中位數），另外以同樣方式量 batch 64 的時間，並換算 samples/s。不使用 CUDA Graphs（選項 c）。
- 理由：使用者詢問建議，Claude 建議 (b)，使用者採用。(1) batch 1 的時間在 4 個寬度都約 4.5 ms，FLOPs 差 6.1 倍卻量不出差別，舊的 H200 量測也有同樣的現象，所以單看 batch 1 無法比較寬度的成本。(2) batch 64 和訓練、val 的 batch 相同，舊的 `bench_cost.py` 也量過，可以和舊的 FM 數字對照（但舊數字是 H200）。(3) CUDA Graphs 雖然能去掉 kernel 啟動的開銷，但 eval 腳本沒有使用它，量到的會是實際評估達不到的速度，Step 6 的推論時間也會對不上。〔出處：Step 2 回報後 Claude 的建議〕
- 被否決的替代方案：(a) 只報 batch 1；(c) 用 CUDA Graphs 量 batch 1。
- 取代：無
- 影響範圍：Step 2 成本表、`scripts/bench_xpred_width.py`、之後的 `results/dit_xpred_width.csv` 成本欄位；Step 6「每幀時間」的算法到時再決定。

### D25　核准進入 Step 3
- 日期：2026-10-27
- 階段：Step 2 的 STOP
- 決定：使用者回覆「繼續」，進入 Step 3（訓練期診斷 logging，以 config 開關控制、預設關閉；以 50 iters 驗證）。D20 第 2、3 步（資料根目錄、real cache）一併在本步完成。
- 理由：（使用者偏好）
- 被否決的替代方案：無（使用者未提出）
- 取代：無
- 影響範圍：Step 3。

### D26　Step 3 診斷設定：N = 1000、非有限值即停止、attention logit 兩種輸入並存
- 日期：2026-10-27
- 階段：Step 3 的 STOP
- 決定：
  (a) R1–R8 的診斷間隔 N = 1000（config `diag.every: 1000`）。
  (b) 遇到第一個 NaN/Inf（grad norm 或 loss）：寫 `nonfinite` 紀錄、存 `nonfinite_step{N}.pt`（該次更新前的權重、EMA、optimizer），
      再寫 `{"stopped": "nonfinite"}` 並以 exit code 3 結束；不做之後的 val、不寫 done.json / last.pt。
      config 開關 `diag.stop_on_nonfinite`（預設 true，R1–R8 明列 true）；diag 關閉時行為與舊版相同。
  (c) max attention logit 同時記兩種輸入：保留原本的 4 個訓練樣本、隨機 t（`attn_max_logit`，與 `--attn-every` 同輸入），
      另加固定 denoising batch（val subset 第一個 batch，64 樣本，固定 noise seed）在每個 denoise t 上的逐層最大值
      （`attn_fixed`，raw 權重，每 16 樣本一塊計算以限制 fp32 logits 的記憶體）。
- 理由：〔D23 補填，出處：2026-10-27 Step 3 STOP 的討論〕
  (a) 使用者原傾向 2000 以與 val 一致；Claude 建議 1000：2000 是 1000 的倍數，每個 val step 仍有對應的診斷紀錄；
      t-bucket loss、grad norm、非有限值計數每步累積，不受 N 影響，N 只決定 denoising error 與 attention logit 的快照密度，
      1000 讓快照加倍（訓練初期尤其有用），額外開銷約 0.4%（每次約 0.61 s）。
  (b) 使用者擔心選繼續會白白浪費算力。Claude 查程式碼：沒有 GradScaler、沒有跳過非有限步的邏輯；grad norm 為 NaN 時
      `clip_grad_norm_` 使所有梯度變 NaN，`opt.step()` 後權重與 Adam 狀態永久變 NaN，EMA 下一步也是，無法恢復；
      繼續跑只會空轉到 50k 步並佔用 QOS 名額。「跳過該步」是另一種介入，會掩蓋不穩定、干擾 v/x 比較，故不採用。
  (c) Claude 的比較：固定輸入讓同一 run 跨步的變化只來自模型（訓練樣本輸入每次紀錄都不同），樣本數 64 對 4，
      且可依 t 拆開；訓練樣本版的優點是反映實際訓練輸入（含 mask 與乘性雜訊）並與 C1/C2 的紀錄可比。
      更正 Step 3 回報：兩種輸入在同一步對 8 個 run 都相同（資料順序、mask、attn_gen seed 與模型無關），跨 run 可比性不是差異所在。
      保留兩者是使用者的選擇（開銷近乎 0）。
- 被否決的替代方案：(a) N = 500、2000；(b) 記錄後繼續訓練、跳過非有限步；(c) 只用其中一種輸入
- 取代：無（Step 3 實作中「記錄後繼續」的行為由本條改為停止；該行為未曾記為決策條目）
- 影響範圍：R1–R8 config 的 `diag` 區塊、`fm/train_fm.py`、`fm/diag.py`（`attn_max_logit_fixed`）、單元測試 (h4)。

### D27　核准進入 Step 4
- 日期：2026-10-27
- 階段：Step 3 的 STOP（D26 實作與驗證後）
- 決定：使用者回覆「繼續」，進入 Step 4：8 個 config 各在 1 張 H100（dev partition）上跑 200 iters 的 smoke test。
- 理由：（使用者偏好）
- 被否決的替代方案：無（使用者未提出）
- 取代：無
- 影響範圍：Step 4。

### D28　定期 checkpoint：每 5000 步存完整 state
- 日期：2026-10-28
- 階段：Step 4 的 STOP
- 決定：R1–R8 每 5000 步存一份 `step{N}.pt`（model、EMA、optimizer，可續訓），共 10 份；config 開關 `train.save_every`（預設 0 = 不存，原行為）。
- 理由：（使用者偏好）
- 被否決的替代方案：每 10000 步（5 份）；只存 model 與 EMA
- 取代：無
- 影響範圍：Step 5；`fm/train_fm.py`、R1–R8 config。估計 8 個 run 共約 100 GB。

### D29　Step 5 排程：8 個 1-GPU job
- 日期：2026-10-28
- 階段：Step 4 的 STOP
- 決定：Step 5 送 8 個單 GPU job（normal partition），受 QOS p_normal 每人 2 個 running job 限制，同時跑 2 個，其餘排隊；估計總牆鐘約 6.5 h。
- 理由：（使用者偏好）
- 被否決的替代方案：1 個 8-GPU job 同時跑 8 個 run（約 2.3 h）
- 取代：無
- 影響範圍：Step 5。

### D30　Nano5 launcher 另開新檔；核准開始 Step 5
- 日期：2026-10-28
- 階段：Step 4 的 STOP
- 決定：(a) 新增 `slurm/fm_nano5.sh` 與 `slurm/gpu_guard_nano5.sh`（account mst114566、partition normal、--mem 96G、不排除 Nano4 節點、用 env 的 python 完整路徑），`slurm/fm.sh`、`gpu_guard.sh` 不動。
  (b) 使用者同意開始 Step 5：R1–R8 各一個單 GPU job（D29），seed 0、50k iters、`train.save_every` 5000（D28）、`diag.every` 1000（D26）。
  送出時的程式碼狀態（HEAD 75d4927 加未 commit 的修改）存於 `results/.../S5_train-50k/code_snapshot/`。
- 理由：(a)〔D23 補填，出處：2026-10-28 Claude 的建議〕任務規則 3 不得改既有預設行為，舊報告的結果都由 `fm.sh` 產生，另開新檔可讓舊腳本維持原樣。(b)（使用者偏好）
- 被否決的替代方案：(a) 直接修改 `fm.sh`
- 取代：無
- 影響範圍：Step 5。

### D31　Step 6 評估設定：rollout 的 K、每幀時間的定義、test NFE
- 日期：2026-10-28
- 階段：Step 5 的 STOP
- 決定：
  (a) N_ar = 10 的 rollout 用 K = 8 條各自獨立的軌跡：每個樣本以自己的預測作為下一步輸入，每一步對 8 條取平均計算指標並報 spread。
  (b) 「每幀時間」主報一次預測呼叫（輸出 20 幀）的時間 = 單次 forward × NFE，並附 ÷ 20 的每個物理幀時間。
  (c) 計時同時報 batch 1 與 batch 64（換算為每個樣本）。
  (d) test 的 NFE 用 Euler 20（Phase 6 的 test 設定）。
- 理由：〔D23 補填，出處：2026-10-28 Claude 的建議〕(a) 最接近「從同一個初始條件生成 8 個可能的未來」，spread 才有意義；每步取平均再接回去會把預測抹平，rollout 只用 K = 1 又和單步設定不一致。(b) 這個資料設定每次呼叫輸出 20 幀，baseline 也一樣，兩種單位都能比較，但 FM 與 baseline 必須一致。(c) batch 1 主要量到 kernel 啟動開銷，4 個寬度都約 4.5 ms（D24），batch 64 才反映寬度的成本。(d) 以便與既有 FM 結果對照。
- 被否決的替代方案：(a) 每步取 8 個樣本的平均再接回去；rollout 只用 K = 1。(b) 只報其中一種單位。(c) 只報一種 batch。(d) val 的 Euler 10。
- 取代：無
- 影響範圍：Step 6。

### D32　散度殘差用 real test 資料的設定，不照任務字面
- 日期：2026-10-28
- 階段：Step 6 前
- 決定：散度殘差不翻 v 的正負號，用資料座標的實際格距（64×128 real 格點，dx ≈ 2.535 mm、dy ≈ 2.729 mm），定義為 ‖∂u/∂x + ∂v/∂y‖₂ / (‖∂u/∂x‖₂ + ‖∂v/∂y‖₂)，interior 去掉邊界 4 格；floor 用同一支程式計算 test 真值本身的值，與預測並列（CLFT Phase 1 的 real floor 為 0.679）。
- 理由：〔D23 補填，出處：2026-10-28 Claude 的建議〕任務寫的 v ← −v、dx = 1.2639 / dy = 1.3551 mm、floor ≈ 0.13 都對應 sim 的 128×256 格點（1.2639/1.3551 = 0.9327 是 sim 格距比，0.13 ≈ sim 128×256 floor 0.1275），而評估用的是 real test（64×128）；real 不需翻號（Stage 2.3：不翻 corr 0.91、翻後 −0.05；CLFT Phase 1 同樣結論）。floor 用同一支程式算 test 真值，預測與 floor 的比較才沒有定義差異。
- 被否決的替代方案：照任務字面計算（翻號、sim 格距、與 0.13 並列）。
- 取代：無（任務第 6 步第 3 項的參數不採用）
- 影響範圍：Step 6 第 3 項。

### D33　圖 3 的 baseline 推論時間在 H100 上重新量測
- 日期：2026-10-28
- 階段：Step 6 前
- 決定：U-Net / DeepONet / Transolver 的推論時間用 Step 6 的計時程式在 H100 上重量（合成 tensor，不需資料），不沿用 todo_1012 B3 的 H200 數字。
- 理由：〔D23 補填，出處：2026-10-28 Claude 的建議〕FM 的 8 個 run 都在 H100 上計時，同一張圖的點要在同一種 GPU 上量才可比。
- 被否決的替代方案：沿用 H200 的 B3 數字並註明。
- 取代：無
- 影響範圍：Step 6 圖 3。

### D34　不另外跑逐位元相同的 K = 1 評估
- 日期：2026-10-28
- 階段：Step 6 前
- 決定：K = 1 的比較用 K = 8 評估中第 1 個樣本的指標（`single` 欄），不另跑單獨的 K = 1。
- 理由：〔D23 補填，出處：2026-10-28 Claude 的建議〕`single` 欄是「隨機取一個樣本」的指標，評估協定和以前的 K = 1 相同，只有第一個 batch 以後的雜訊序列不同，不會逐位元一樣。
- 被否決的替代方案：另跑一次 K = 1，和以前的評估逐位元相同。
- 取代：無
- 影響範圍：Step 6。

### D35　N_ar = 10 只評估 FM 的 8 個 run，metric_batch_size = 64（D15 的決定）
- 日期：2026-10-28
- 階段：Step 6 前
- 決定：N_ar = 10 的 rollout 只跑 R1–R8，不補跑 baseline；指標依官方規則（N_ar > 4 時 metric_batch_size = test_batch_size），8 個 run 一律用 test_batch_size = 64。每一輪的 Rel L2 逐筆計算後對全部樣本平均，不分塊。
- 理由：〔D23 補填，出處：2026-10-28 Claude 的建議〕圖 2 只需要 8 條 FM 的線；8 個 run 用同一個塊大小，彼此可比，也符合官方 N_ar > 4 的規則。paper §4.5 與圖 3c 的 cylinder 自迴歸評估最多 10 輪，與任務的 N_ar = 10 一致；官方 cylinder config 的 N_autoregressive 只有 1、3、5、10。
- 被否決的替代方案：baseline 也補跑 N_ar = 10（各用自己的 test_batch_size，或全部統一塊大小）；照 paper 跑 1、2、3、5、10 輪。
- 取代：無（D15 延後的事項在此決定）
- 影響範圍：Step 6 第 2 項、圖 2。

### D36　N_ar = 10 的 rollout 改用 K = 1
- 日期：2026-10-28
- 階段：Step 6（正式評估前）
- 決定：N_ar = 10 的 rollout 每個輸入只生成 1 條軌跡（K = 1），報每一輪的 Rel L2；單步（N_ar = 1）仍為 K = 8，含 ensemble mean 與 spread（D9）。使用者同意開始正式評估。
- 理由：rollout 用 K = 8 佔總評估成本約九成（約 21 / 24 GPU-h）；任務原文只要求單步用 K = 8，rollout 未指定。使用者的原則：在進入我們自己設計的方法之前，小規模實驗看趨勢即可，不過度消耗算力，必要的實驗仍要做。
- 被否決的替代方案：維持 D31(a)（K = 8 條獨立軌跡，約 24 GPU-h）。
- 取代：D31(a)
- 影響範圍：Step 6 第 2 項、圖 2（rollout 沒有 spread）。總成本約 5 GPU-h。
