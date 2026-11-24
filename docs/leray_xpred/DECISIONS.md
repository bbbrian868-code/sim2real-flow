# 決策紀錄：Leray 投影（散度為 0）作為推論期硬約束的對照組

任務：`docs/leray_xpred/TASK_2026-11-22_leray-xpred.md`；結果檔：`/work/b314513067/pi-lfm/results/fm/2026-11-22_leray-xpred/`。
本檔只記錄這個實驗的決策；前一個實驗（PCFM、約束 C2）的決策在 `docs/pcfm_xpred/DECISIONS.md`（P1–P31），`dit_xpred_width` 的在 `docs/dit_xpred_width/DECISIONS.md`，本 repo 更早的決策在 `docs/DECISIONS.md`。

每筆的格式：編號、日期、步驟、決定、理由。理由欄空白表示使用者偏好。推翻時新增一筆，並在決定欄註明取代的編號。

| # | 日期 | Step | 決定 | 理由 |
|---|---|---|---|---|
| L1 | 11-22 | 規劃 | 新增「散度為 0」對照：以逐點線性約束 $D\mathbf{u}=0$ 的 Leray 投影（方法 B：遮罩、非週期、矩陣自由 CG）實作；不以 C2 設 τ = 0 | div_rel 的範數在 0 處不可微，Gauss–Newton 的收斂假設失效；Leray 投影是線性約束的精確最小改動解 |
| L2 | 11-22 | 規劃 | 方法 A（FFT）只用於驗證方法 B，不作為實驗組 | cylinder 非週期且有障礙物，FFT 會讓邊界互相繞接 |
| L3 | 11-22 | 規劃 | 只跑反推噪聲退回（`retreat=implied`），不再跑初始噪聲退回 | P30–P31 已顯示初始噪聲退回本身就讓誤差變大；反推版本不加約束時等於 Euler，約束的效果才能被分離出來 |
| L4 | 11-22 | 規劃 | Leray 的實驗組為 {存檔尺度 D_raw, 尺度修正 D_s} × {無壁面條件 free, 壁面條件 wall} × {只在最後, 每步} 共 8 組；另加部分投影 P-raw 1 組（沿 D_raw、free 的 Leray 方向，只在最後，目標 τ_last） |  |
| L5 | 11-22 | 規劃 | 加入真值 oracle：對 real test 與 sim test 的真值做 Leray 投影，報告投影前後的指標差與 div_rel；real 做 4 種（D_raw / D_s × free / wall），sim 做 2 種（D_raw × free / wall） | 量出「約束本身的代價」，與模型無關；sim 與 real 並列可量化同一條約束在兩種資料上的代價差 |
| L6 | 11-22 | 規劃 | D_s 的尺度：$s=1/r$，$r$ 為每條 trajectory 的 CLFT β/α（`vu-scale-and-rerank` 報告 §2 的 v/r 變體所用的值）；sim 一律 $s=1$ | 三種獨立估計都指向 v/u 約大 2.8 倍；用與既有「v/r」評估變體相同的逐 trajectory 值，數字可以和既有表格對照 |
| L7 | 11-22 | 規劃 | 壁面條件版本（wall）：圓柱內部格點的 u、v 設為 0 且不再被投影修改；散度約束區從 Ω 擴大為 Ω_wall（Ω 加上圓柱周圍、因膨脹而被排除的流體格點，確切定義依 Step 0 的盤點，由我確認）；PIV 無效格點仍排除 | 只覆寫圓柱內部而不擴大約束區，差分不會碰到圓柱格點，條件對 Ω 內的解沒有作用 |
| L8 | 11-22 | 規劃 | 所有組別的指標同時報：官方（全網格）、排除圓柱內部、strict Ω；以及既有的 v/r 評估變體 | 壁面條件會在圓柱內部（真值 0.48 U∞）計入誤差；v/u 尺度問題讓 v 的權重偏大 |
| L9 | 11-22 | 規劃 | 模型：R6（real 訓練，`best.pt` step 50000）與 R6num（sim 訓練，`best.pt` step 4000，以 real val 選點，屬上限，P28）都跑，不指定主模型 |  |
| L10 | 11-22 | 規劃 | $N_{ar}=1$（K = 8）與 $N_{ar}=10$（K = 1）都跑；test noise seed 1234、val seed 0（同 P12）；C2 組在 $N_{ar}=10$ 時每輪的目標值沿用真實輸入最後一幀（同 P25）；Leray 組不需要目標值 |  |
| L11 | 11-22 | 規劃 | 投影在存檔的物理單位下做（反正規化 → 投影 → 再正規化，統計量同 `DivRelConstraint`：`mean_targets` / `std_targets`）；壓力通道不動；CG 以 fp64 計算，相對容忍值預設 1e-10、上限 5000 次，Step 3 量測後再確認 | 正規化空間的散度沒有物理意義；fp64 的額外成本相對於一次 DiT forward 可忽略 |
| L12 | 11-22 | 規劃 | 評估協定同 P7、P20–P22：real test 4827 筆（$N_{ar}=1$）/ 4672 筆（$N_{ar}=10$），Euler 20 步，fp32，EMA，官方評估迴圈；C2 組的最終 Newton 投影設定同 P17、P22 | 與 `pcfm_xpred` 的所有數字可直接並列 |
| L13 | 11-22 | 規劃 | R6 已有的 A0、A2'、A3'（S6、S7、S8）不重跑：先在前 128 筆 test 上確認新程式以相同參數重現（$N_{ar}=1$ 逐位元或列出差異來源；$N_{ar}=10$ 相對差 ≤ 1e-3，同 P31），再沿用既有結果 |  |
| L14 | 11-22 | 規劃 | R6num 不改用 sim val 重新選 checkpoint，維持 L9 的 `best.pt`（step 4000），所有 R6num 的數字標註為上限（P28） |  |
| L15 | 11-22 | 規劃 | 不做 R6num 在 sim test 上的模型評估；sim test 只用於 Step 2 的真值 oracle |  |
| L16 | 11-22 | Step 0 的 STOP | sim 的散度以物理方向的 v 計算：sim 的 s = −1（存檔的 v 與物理方向相反，D-D03；存檔 v 算出的 div_rel 約 0.99，反號後約 0.12）（取代 L6 中「sim 一律 s = 1」的部分） |  |
| L17 | 11-22 | Step 0 的 STOP | 圓柱本體 = 幾何圓盤（半徑 D/2，Phase D 的中心），real 與 sim 相同；wall 版本固定的格點 = 圓盤，Ω_wall = 離邊界 4 格以內除外、扣掉圓盤、扣掉 PIV 無效格點（= Ω 加上膨脹環中的流體格點）（L7 的確切定義） |  |
| L18 | 11-22 | Step 0 的 STOP | 投影允許修改約束區外圍、Dᵀ 所及的格點（膨脹環、邊界帶、PIV 無效格點）；只有 wall 版本的圓柱本體固定為 0 |  |
| L19 | 11-22 | Step 0 的 STOP | 實作方式：新增 `fm/leray.py`；`constrained_sample` 以約束物件是否提供 `project` 決定走封閉解（不看 `linear`）；`eval_xpred_width.py` 新增 `--constraint c2\|leray\|partial`（預設 c2）、`--div-scale raw\|vr`、`--wall`，新組別的輸出檔名加後綴、C2 組維持原檔名；另寫新程式產生含 strict Ω、圓柱本體、Ω_wall、r 的幾何檔（real、sim 各一），舊檔不動；val 用的 smoke 另寫新程式，不改 `pcfm_xpred_smoke.py` |  |
| L20 | 11-22 | Step 1 的 STOP | L8 的「排除圓柱內部」以排除幾何圓盤（半徑 D/2，同 L17 的圓柱本體）為主要報告；排除膨脹障礙物（背景所說約 2.4% 格點、u 平均 0.48 U∞ 的區域）放附表 |  |
| L21 | 11-22 | Step 1 的 STOP | 部分投影（P-raw）在搜尋範圍內無解的幀：該幀不修改（α = 0），並記錄為無解 |  |
| L22 | 11-22 | Step 1 的 STOP | R6 的 A0、A2'、A3'（$N_{ar}=1$）在 Step 4 以相同參數加上區域指標重跑，結果需與既有結果相同（能譜的末位誤差除外）（取代 L13 中 $N_{ar}=1$「不重跑」的部分；$N_{ar}=10$ 仍依 L13） |  |
| L23 | 11-22 | Step 1 的 STOP | Step 3 的 val smoke 另寫新程式，不修改 `pcfm_xpred_smoke.py` |  |
| L24 | 11-22 | Step 3 的 STOP | Step 4 照原計畫全部跑：R6 的 8 個 Leray 組 + P-raw（加 L22 的重跑），R6num 的 A2'、A3' + 8 個 Leray 組 + P-raw；每步投影組的中間步驟不放寬 CG 容忍值 |  |
| L25 | 11-22 | Step 3 的 STOP | CG 相對容忍值 1e-10、上限 5000 次（L11 的確認），最後一次投影與每步投影都用此設定 |  |
| L26 | 11-22 | Step 3 的 STOP | R6num 的 A0（$N_{ar}=1$）在 Step 4 以相同參數加上區域指標重跑，結果需與既有結果（`pcfm_xpred` S6）相同（能譜的末位誤差除外） |  |
| L27 | 11-23 | Step 4 的 STOP | Step 5 照原計畫開始：先做前 128 筆 test 的重現檢查，通過後跑 R6 的 8 個 Leray 組 + P-raw、R6num 的 A0、A2'、A3' + 8 個 Leray 組 + P-raw |  |
| L28 | 11-23 | Step 4 的 STOP | 兩個模型的 A0 與 R6 的 A2'、A3'（$N_{ar}=1$）的推論時間，以其他組相同的量測方式（前 64 筆 real test 輸入，batch 1 與 64）重新量測，取代 Step 4 總表中借用的時間；`eval_xpred_width.py bench` 在給 `--geometry` 時，Euler 也走這個量測方式 |  |
| L29 | 11-23 | Step 5 | R6 的 A0、A2'、A3'（$N_{ar}=10$）以相同參數加上區域指標重跑（A0 另加 `--omega` 與 C2 殘差，取得 div_rel），結果需與既有結果（`dit_xpred_width` S6、`pcfm_xpred` S8）相同（能譜的末位誤差除外）（取代 L13 中 $N_{ar}=10$「不重跑」的部分） |  |
| L30 | 11-24 | Step 5 的 STOP | L29 的 A0 重跑比對另以跳過 `omega_checks` 的比對程式重做（原比對程式與結果保留）；`omega_checks` 只因重跑加了 `--omega` 而由空值變為檢查值 |  |
| L31 | 11-24 | Step 5 的 STOP | 官方 200 幀 Rel L2 與區域指標「全網格」的差異（$N_{ar}=10$ 最多 2.9e-5，每輪值相同）只記錄，不追查 |  |
| L32 | 11-24 | Step 5 的 STOP | 寫結果總整文件，連同程式、DECISIONS 與總表 CSV 一起 commit，推到分支 |  |
