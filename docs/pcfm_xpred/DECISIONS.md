# 決策紀錄：在 x-pred FM 上實作 PCFM（推論期硬約束取樣）

任務：`docs/pcfm_xpred/TASK_2026-11-08_pcfm-xpred.md`；結果總整：`docs/pcfm_xpred/RESULTS_2026-11-09_pcfm-xpred.md`；結果檔：`/work/b314513067/pi-lfm/results/fm/2026-11-08_pcfm-xpred/`。
本檔只記錄這個實驗的決策；`dit_xpred_width` 的決策在 `docs/dit_xpred_width/DECISIONS.md`（D1–D39），本 repo 更早的決策在 `docs/DECISIONS.md`。

每筆的格式：編號、日期、步驟、決定、理由。理由欄空白表示使用者偏好。推翻時新增一筆，並在決定欄註明取代的編號。

| # | 日期 | Step | 決定 | 理由 |
|---|---|---|---|---|
| P1 | 11-08 | 規劃 | 只做推論，不訓練、不微調；使用 `dit_xpred_width` 的 x-pred checkpoint（EMA、val 選出的 `best.pt`） | PCFM 與 ECI 都是 zero-shot 方法，重點是同一個無條件模型在推論時能否施加約束 |
| P2 | 11-08 | 規劃 | 只支援 `pred: x` 的模型；遇到 v-pred 模型直接報錯 | x-pred 的外推就是網路輸出，不需經過速度換算；v-pred 模型在 width < 1024 時本身就有容量問題（`dit_xpred_width` RESULTS §4） |
| P3 | 11-08 | 規劃 | Relaxed correction 依官方實作的形式（速度場固定為當前步的 v、不對網路反傳；見任務檔「參考實作」第 5 點）以開關加入，預設關閉（λ = 0）；是否在 Step 3 / 4 開啟另行決定 | 論文 Appendix H 的預設是 λ = 0；官方實作的形式只需要 ∇h，成本可接受，保留作為可選項 |
| P4 | 11-08 | 規劃 | 每一步的退回使用該樣本的初始噪聲 y0，全程固定；A0–A3 使用同一個 generator 與相同的抽取順序，使各取樣方式的樣本以噪聲配對 | 配對後，各方式之間的差異只來自取樣器與約束 |
| P5 | 11-08 | 規劃 | 約束一律在物理單位下計算（反正規化後）；壓力通道 p 不施加約束，除非 Step 2 另有決定 | 正規化空間中 u、v 的 std 不同，算出的散度不是物理散度；官方指標只計算 u、v |
| P6 | 11-08 | 規劃 | 不施加逐點的 ∇·u = 0 | 約束數 m ≈ n，JJᵀ 無法求解；real test 真值本身的 div_rel 中位數約 0.68（RESULTS §4.2），逐點零散度與資料不符 |
| P7 | 11-08 | 規劃 | 評估協定與 `dit_xpred_width` Step 6 相同：real test，Euler N = 20，fp32，EMA，N_ar = 1 時 K = 8，noise seed 0，div_rel 的定義與 Ω、格距依 D32（`scripts/xpred_width_omega.py` 產生的 OMEGA.npz） | 與既有 FM 及 baseline 的數字可直接並列 |
| P8 | 11-08 | 規劃 | 投影求解器的數值精度：fp32 起步；若 JJᵀ 的條件數過大導致收斂失敗，回報後再決定是否改用 fp64 | 先量測，不預設 |
| P9 | 11-08 | 規劃 | 中間步驟投影的 Tikhonov ε 以參數加入，預設 1e-6（同官方 `pcfm_sample`）；最終投影預設 ε = 0，迭代到 ‖h‖ < ε_tol 或達到 `newton_max_iter`（依論文 Theorem A.1，不依官方的 `max_iter=1`） | 與官方中間步驟的數值行為一致；最終投影依論文才能保證約束精確成立；實際迭代次數在 Step 3 量測回報 |
| P10 | 11-08 | Step 0 前 | 約束只考慮 C1（障礙物 / 無效格點的指定數值）與 C2（每幀 div_rel 等於目標值 τ）；不做 C3（被 P11 取代） |  |
| P11 | 11-08 | Step 0 的 STOP | 約束只做 C2（每幀 div_rel 等於目標值 τ，m = 20，非線性）；不做 C1、C3；因此不做 A4（ECI 只適用於線性約束）（取代 P10） |  |
| P12 | 11-08 | Step 0 的 STOP | Step 4 的 noise seed 用 1234（S6 實際使用的值，`eval_xpred_width.py` 的預設），A0 直接沿用 S6 的結果、A1–A3 與之以噪聲配對；Step 3 在 val 上仍用 seed 0（取代 P7 中「noise seed 0」的部分） |  |
| P13 | 11-08 | Step 0 的 STOP | 主要模型用 R6（x-pred、width 512，`best.pt` step 50000） |  |
| P14 | 11-08 | Step 0 的 STOP | 進入 Step 1（實作取樣器與約束介面，不跑任何模型推論） |  |
| P15 | 11-08 | Step 1 | J 的計算方式預設 `jac="jacrev"`（vmap(jacrev)，J 實體化）；`jac="vjp"`（每個約束一次 vjp + 一次 jvp 組出 JJᵀ，不存 J）保留為選項。C2、B = 64、n = 491,520 一次投影實測：jacrev 49 ms、峰值 +7.3 GiB；vjp 96 ms、峰值 +0.84 GiB；兩者結果最大差 1.8e-6 |  |
| P16 | 11-08 | Step 1 的 STOP | 中間步驟投影的 Tikhonov ε 維持 1e-6（同官方），不依資料尺度調整（P9 不變） |  |
| P17 | 11-08 | Step 1 的 STOP | 最終 Newton 投影最多 20 次（`newton_max_iter = 20`）；停止容忍值暫定 ‖h‖ < 1e-5，Step 3 量測後再確認；20 次仍未達到的樣本標記為未收斂 |  |
| P18 | 11-08 | Step 1 的 STOP | 「只在最後投影」（A3）若最終投影不收斂，照實回報不收斂的比例與殘差，不另做備案 |  |
| P19 | 11-08 | Step 2 的 STOP | 約束的目標值逐樣本設定：每筆樣本 20 個輸出幀的 div_rel 目標都等於該樣本輸入最後一幀的 div_rel（D32 定義，只用到預測時已知的輸入）；選定方式依 val 上的比較（Step 2，`S2_div-target-stats/`），正式評估時由 test 樣本自己的輸入幀算出 |  |
| P20 | 11-08 | Step 3 的 STOP | 進行 Step 4：A1–A3 在 real test 上跑 N_ar = 1、K = 8、noise seed 1234（P12）；A0 先在 128 筆 test 上確認新程式重現 S6，再沿用 S6 的完整結果 |  |
| P21 | 11-08 | Step 3 的 STOP | Step 4 的實作方式：在 `scripts/eval_xpred_width.py` 以參數開關加入取樣器選擇，預設值維持原行為；開啟時每批先算出各樣本的目標值與 Ω、格距再交給取樣器 |  |
| P22 | 11-08 | Step 3 的 STOP | 最終 Newton 投影的停止容忍值確定為 ‖h‖ < 1e-5（確認 P17 的暫定值） |  |
| P23 | 11-08 | Step 3 的 STOP | 不加入「退回時改用由當前狀態反推的噪聲」的變體；退回維持使用初始噪聲（P4 不變） |  |
| P24 | 11-09 | Step 4 的 STOP | 加跑 10 輪自回歸 rollout：A1–A3 在 real test 上跑 N_ar = 10、K = 1、noise seed 1234（同 S6 的 A0 設定）；A0 先在前 128 筆 test 上確認新程式重現原程式，再沿用 S6 的完整結果；A0 的約束殘差只在這 128 筆上算 |  |
| P25 | 11-09 | Step 4 的 STOP | 10 輪 rollout 中每一輪的目標值都沿用該樣本真實輸入最後一幀的 div_rel（第 2–10 輪不以回饋的預測幀重算） |  |
| P26 | 11-09 | Step 4 的 STOP | 檢查評估重算殘差與取樣器內部殘差的差距：在前 64 筆 val 上，把同一份輸出依序加上 fp64/fp32 計算、反正規化、官方迴圈的正規化往返，逐項量殘差（`S5_residual-gap-val/`） |  |
| P27 | 11-16 | 延伸：sim 訓練模型 | 加做 sim-to-real：用 R6 的同一份 config（`R6_xpred_w512.yaml`），只把訓練資料改成 numerical（`--train-data-type numerical`，seed 0，50k 步，H100），訓練一個 x-pred 模型 R6num；在 real test 上跑 A0–A3，N_ar = 1、K = 8、noise seed 1234，其餘同 P20–P23。只跑 1 個 seed，不做 finetune，不做 N_ar = 10 | 只用 real 訓練的模型測不到 sim-to-real。模擬資料的 div_rel 約 0.12，real 約 0.68；模擬訓練的模型可能輸出散度偏低的流場，約束的目標值來自 real 輸入，可能有落差可以修正。R6 上約束沒有幫助，可能是因為模型已經學到 real 的統計。先用 1 個 seed 看趨勢，成本約 3 GPU-h |
| P28 | 11-16 | 延伸：sim 訓練模型 | R6num 的 checkpoint 選點沿用 `train_fm.py` 的做法：每 2000 步在 536 筆 real val 子集上算 RMSE，取最低的 `best.pt`；`last.pt` 也保留 | 與 R6、Phase 6 的 numerical 列及官方流程相同，可以直接比較。D-031 指出：模擬訓練用 real val 選點會用到 real 資料，得到的是上限，結果中需註明 |
| P29 | 11-16 | 延伸：sim 訓練模型 | 建立 numerical cache（`scripts/build_cache.py --type numerical` → `pi-lfm/cache/v2.0.1/numerical`），以單元測試 (f) 驗證和官方 loader 逐位元相同；不寫入 clft_data | config 的 `cache_dir` 需要它，D20 當時只重建了 real cache；驗證方式和 real cache 相同 |
| P30 | 11-18 | 延伸：反推噪聲的退回 | 加做「退回時改用由當前狀態反推的噪聲」的變體（取代 P23；P4 只在 `--retreat init`，即預設值下成立）：每一步以當前狀態與網路原始輸出反推噪聲 (y_t − t·x)/(1 − t)，再與（投影後的）x 內插到下一個時間點。不加約束時這和原本的 Euler 取樣代數上相同，所以先在前 128 筆 test 上確認它重現 A0，再在 real test 上跑「每步投影」與「只在最後投影」兩種，R6、N_ar = 1、K = 8、noise seed 1234，其餘同 P20–P23（`S7_implied-noise-retreat/`，約 0.6 GPU-h）；不做 N_ar = 10、不加 relaxed correction |  |
| P31 | 11-19 | 延伸：反推噪聲的退回 | 反推噪聲的兩種取樣方式（每步投影、只在最後投影）加跑 10 輪自回歸 rollout：R6、real test、N_ar = 10、K = 1、noise seed 1234，每一輪的目標值沿用真實輸入最後一幀的 div_rel（同 P24、P25）；先在前 128 筆 test 上確認不加約束的反推版本重現 A0 的 10 輪結果（容許相對差 1e-3），再跑正式評估（`S8_implied-noise-rollout10-test/`，約 1.1 GPU-h） |  |
