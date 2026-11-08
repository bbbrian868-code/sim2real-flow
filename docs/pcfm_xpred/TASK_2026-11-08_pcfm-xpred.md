# Task: 在 x-pred FM 上實作 PCFM（推論期硬約束取樣）

> 本檔是給 Claude Code 的執行清單。
> 依據：Utkarsh et al., *Physics-Constrained Flow Matching*, NeurIPS 2025（PCFM, arXiv:2506.04171）；Cheng et al., *Gradient-Free Generation for Hard-Constrained Systems*, ICLR 2025（ECI, arXiv:2412.01786）。
> 本實驗**只做推論，不訓練任何模型**；使用 `dit_xpred_width` 實驗已訓練好的 x-pred checkpoint（R5–R8）。

---

## 目標

在不重新訓練的前提下，把 PCFM 取樣器接到現有的 x-pred rectified-flow 模型上，使生成結果在數值精度內滿足指定的等式約束 $h(\hat y)=0$，並在 real test 上以與 `dit_xpred_width` Step 6 相同的評估協定，比較以下取樣方式：

| 代號 | 取樣方式 | 用途 |
|---|---|---|
| A0 | 原本的 Euler（`fm/paths.euler_sample`） | 參考 |
| A1 | 「外推 → 退回」取樣器，**不施加約束**（修正步驟為 identity） | 隔離「取樣器本身的改變」造成的差異 |
| A2 | PCFM：每步做一次 Gauss–Newton 投影，最後做完整 Newton 投影 | 主方法 |
| A3 | 只在最後投影（中間步驟不投影） | 對照 PCFM 論文 Table 6 的 ablation |
| A4 | ECI（僅限線性約束時；即 A2 在線性約束下的特例，另含 mixing $M$ 與噪聲重抽 $R$） | 僅在 Step 2 選定的約束包含線性約束時才做 |

---

## 背景：演算法與本 codebase 的對應

本 codebase 的慣例（`fm/paths.py`、`fm/build.py`）：$t=0$ 為噪聲、$t=1$ 為資料，$y_t=(1-t)\,y_0+t\,y_1$；x-pred 模型為 `XPredVelocity(net, t_floor)`，`net(y_t, t, cond)` 直接輸出 $\hat y_1$，速度為 $(\hat y_1-y_t)/\max(1-t,\,t_\text{clip})$。

PCFM 每個 Euler 步（$t_k=k/N$，$t_{k+1}=(k+1)/N$）：

```
1. 外推      ŷ1     = net(y_t, t_k, cond)                       # x-pred 直接輸出，不經速度換算
2. 投影      r = h(ŷ1),  J = ∂h/∂ŷ1
             ŷ1_proj = ŷ1 − Jᵀ (J Jᵀ)⁻¹ r                        # 一次 Gauss–Newton（切空間正交投影）
3. 退回      y_{t_{k+1}} = (1 − t_{k+1}) · y0 + t_{k+1} · ŷ1_proj  # y0 = 本樣本的初始噪聲，全程固定
```

全部步驟結束後（此時 $y = \hat y_{1,\text{proj}}$ 的最後一次結果），若 $\|h(y)\|>\varepsilon$，做以 $y$ 為錨點的 Newton–Schur 迭代（PCFM Eq. 34）：

$$
u^{(j+1)} = a - J_j^\top (J_jJ_j^\top)^{-1}\big(h(u^{(j)}) + J_j\,(a-u^{(j)})\big),\qquad J_j=\nabla h(u^{(j)}),\ a = y
$$

- NFE 與 Euler 相同（每步一次 `net` forward），網路永遠不在 $t=1$ 被呼叫。
- 只需要 $\nabla h$，**不對網路反傳梯度**。
- 約束為線性（$h=Ay-b$）時，第 2 步一次即精確，且等於 ECI 的修正步驟。
- $h$ 必須在**物理單位**下計算：網路輸出是正規化後的 `(B, T_out·C, H, W)`，需經 `unfold` 與官方 `GaussianNormalizer` 反正規化後才計算物理量。Jacobian 透過 autodiff 自動包含反正規化這一層。

---

## 參考實作（唯讀參考，不得直接 import 或 vendoring）

| repo | commit | 相關檔案 |
|---|---|---|
| https://github.com/cpfpengfei/PCFM （MIT；含源自 ECI 的 Apache-2.0 元件） | `306c82e` | `pcfm/pcfm_sampling.py`（`pcfm_sample`、`compute_jacobian`、`fast_project_batched`、`relaxed_penalty_constraint_interp_linear_detached`）、`pcfm/ffm_sampler.py`（`FFM_sampler.pcfm_sample`、`FFM_NS_sampler.pcfm_sample`、`eci_sample`）、`pcfm/constraints.py`（各 PDE 的殘差函數） |
| https://github.com/amazon-science/ECI-sampling （Apache-2.0） | `8c3061a` | `models/functional.py`（`FFM.eci_sample`） |

**已知的論文與官方程式碼差異**（實作時以本任務檔的規定為準；Step 0 須逐項確認並回報是否還有其他差異）：

1. **介面錯誤**：`ffm_sampler.py` 呼叫 `fast_project_batched(u, hfunc, mode=..., newtonsteps=..., eps=..., use_vmap=...)`，但 `pcfm_sampling.py` 中該函式的簽名只有 `(xi_batch, h_func, max_iter=1)`，照原樣執行會丟出 `TypeError`。官方 sampler 的最終投影因此無法直接作為參考實作執行。
2. **最終投影的停止條件**：論文寫「若 $\|h\|>\varepsilon$ 則迭代至收斂」（Theorem A.1）；`fast_project_batched` 預設只做 `max_iter=1` 次 Newton，且沒有容忍值檢查。
3. **Tikhonov 正則化**：中間步驟的投影（`pcfm_sample`）解的是 $(JJ^\top+\epsilon I)\lambda=r$，$\epsilon=10^{-6}$；最終投影沒有加 $\epsilon$。論文沒有提到 $\epsilon$。
4. **Relaxed correction 預設開啟**：`FFM_sampler.pcfm_sample` 的 `guided_interpolation` 預設 `True`；論文 Appendix H 說預設 $\lambda=0$、只有 Heat 開啟。
5. **Relaxed correction 的速度場是固定的**：論文 Eq. 6 寫 $h(u+\gamma\,v_\theta(u,\tau'))$，官方實作用的是當前狀態算出的 `v_flat`（已 detach），在梯度下降中不重新呼叫網路，所以**不需要對網路反傳**；另外 $\gamma=\max(1-\tau',\,10^{-3})$，以固定步長梯度下降（預設 20 步、step size $10^{-2}$、$\lambda=1$）。
6. **推進方式**：`pcfm_sample` 回傳修正後的速度 `(ut_interp − u_t)/dt`，外層再做 `u ← u + dt·v_proj`，數學上等於直接令 $u=\hat u_{t'}$（只差浮點捨入）。
7. **2D 不使用 vmap**：`pcfm_2d_batched` 為了避免 OOM 改為逐樣本 Python 迴圈，每個樣本對完整的 $n$ 維向量做 `jacrev`。
8. **ECI 的 $R$**：以 ECI step 的次數計數（每次 mixing 都加一），不是以 Euler 步計數；`None` 或 0 表示不重抽。

---

## 全域規則（每一步都適用）

1. **每完成一個 Step 就停下回報**，等我回覆「繼續」才進下一步。
2. **未經明確同意，不得啟動任何正式推論**（smoke test 僅限 Step 3 規定的範圍）。本任務不得啟動任何訓練。
3. **不得修改既有預設行為**：所有新功能以新檔案或 config / 參數開關加入，預設值必須重現舊行為（`FMPredictor`、`euler_sample`、`eval_xpred_width.py` 的既有呼叫結果不變）。
4. **遇到文件或程式碼語意不明時，不得自行假設**，列出選項並回報。
5. **發現架構或設定前後不一致時，直接標出來**，不要自行做局部修補。
6. **不得使用 test split 決定任何東西**：checkpoint、超參數、約束的目標值（例如散度殘差的 target）都只能由 train 或 val 決定。
7. 只回報數字與觀察到的現象，不要替結果下研究結論。
8. **維護決策紀錄** `docs/pcfm_xpred/DECISIONS.md`（格式沿用根目錄 `DECISIONS.md` 的表格：編號、日期、Step、決定、理由；理由欄空白表示使用者偏好；推翻時新增一筆並在決定欄註明取代的編號）。
   - Step 0 開始前建立此檔，並把下方「已確定的決策」P1–P9 寫入。
   - 理由必須是我給的理由，不得自行代填；我沒說明理由時理由欄留空。
   - 每步回報的最後附上本步新增的決策編號。
9. **參考實作只能閱讀，不得直接 import 或複製進 repo**。若有任何函式是改寫自參考實作，須在檔頭註明來源（repo、commit、函式名），並依本 repo 的規定更新 `NOTICE`（列出改寫的檔案與修改內容）；授權條款相容性有疑慮時先回報。
10. 本任務檔原樣存為 `docs/pcfm_xpred/TASK_2026-11-08_pcfm-xpred.md`；結果根目錄 `/work/b314513067/pi-lfm/results/fm/2026-11-08_pcfm-xpred/`，子目錄命名沿用 `S{step}_{描述}/`。

---

## 已確定的決策

| # | 決定 | 理由 |
|---|---|---|
| P1 | 只做推論，不訓練、不微調；使用 `dit_xpred_width` 的 x-pred checkpoint（EMA、val 選出的 `best.pt`） | PCFM 與 ECI 都是 zero-shot 方法，重點是同一個無條件模型在推論時能否施加約束 |
| P2 | 只支援 `pred: x` 的模型；遇到 v-pred 模型直接報錯 | x-pred 的外推就是網路輸出，不需經過速度換算；v-pred 模型在 width < 1024 時本身就有容量問題（`dit_xpred_width` RESULTS §4） |
| P3 | Relaxed correction 依官方實作的形式（速度場固定為當前步的 $v$、不對網路反傳；見「參考實作」第 5 點）以開關加入，**預設關閉**（$\lambda=0$）；是否在 Step 3 / 4 開啟由我決定 | 論文 Appendix H 的預設是 $\lambda=0$；官方實作的形式只需要 $\nabla h$，成本可接受，保留作為可選項 |
| P4 | 每一步的退回使用該樣本的初始噪聲 $y_0$，全程固定；A0–A3 使用同一個 generator 與相同的抽取順序，使各取樣方式的樣本以噪聲配對 | 配對後，各方式之間的差異只來自取樣器與約束 |
| P5 | 約束一律在物理單位下計算（反正規化後）；壓力通道 $p$ 不施加約束，除非 Step 2 另有決定 | 正規化空間中 $u$、$v$ 的 std 不同，算出的散度不是物理散度；官方指標只計算 $u$、$v$ |
| P6 | 不施加逐點的 $\nabla\cdot\mathbf{u}=0$ | 約束數 $m\approx n$，$JJ^\top$ 無法求解；real test 真值本身的 div_rel 中位數約 0.68（RESULTS §4.2），逐點零散度與資料不符 |
| P7 | 評估協定與 `dit_xpred_width` Step 6 相同：real test，Euler $N=20$，fp32，EMA，$N_{ar}=1$ 時 $K=8$，noise seed 0，div_rel 的定義與 Ω、格距依 D32（`scripts/xpred_width_omega.py` 產生的 OMEGA.npz） | 與既有 FM 及 baseline 的數字可直接並列 |
| P8 | 投影求解器的數值精度：fp32 起步；若 $JJ^\top$ 的條件數過大導致收斂失敗，回報後再決定是否改用 fp64 | 先量測，不預設 |
| P9 | 中間步驟投影的 Tikhonov $\epsilon$ 以參數加入，預設 $10^{-6}$（同官方 `pcfm_sample`）；最終投影預設 $\epsilon=0$，迭代到 $\|h\|<\varepsilon_\text{tol}$ 或達到 `newton_max_iter`（依論文 Theorem A.1，不依官方的 `max_iter=1`） | 與官方中間步驟的數值行為一致；最終投影依論文才能保證約束精確成立；實際迭代次數在 Step 3 量測回報 |

---

## Step 0　盤點（唯讀，不改任何檔案）

填寫下表並回報：

| 項目 | 值 | 出處（檔案:行號） |
|---|---|---|
| 候選 checkpoint（R5–R8）的路徑、`best.pt` 的步數、config 中的 `pred`、`target_clip`、`t_clip` / `t_floor` | | |
| 取得 $\hat y_1$ 的方式：`XPredVelocity.net` 的呼叫簽名；確認 `y_t + max(1−t, t_clip)·v` 與 `net(...)` 逐位元相同或列出差異 | | |
| 網路輸出的 tensor 形狀與通道排列（`fold` 後第 $t\cdot C + c$ 個通道對應哪一幀、哪個變數） | | |
| `GaussianNormalizer` 的統計量形狀（逐通道？逐像素？）、反正規化的呼叫方式；`mean_std.pt` 的 sha256 | | |
| real test 中每個 sim 的 Ω、dx、dy 的來源與形狀；Ω 在同一 batch 內是否因 sim 而不同 | | |
| 障礙物區域與 PIV 無效格點在 real 資料中的實際數值（物理單位與正規化後）：是否為定值、是否隨時間改變、是否每個 sim 相同 | | |
| `eval_xpred_width.py` 中可以接入新取樣器的位置（`make_draw` 或 `FMPredictor.draw`）；noise generator 的抽取順序 | | |
| `div_rel` 是否可微（有無 in-place 操作、布林索引）；若要在 batch 中逐樣本使用不同 Ω，需要怎麼改寫才能配合 `torch.func.vmap` | | |
| `dit_xpred_width` 中各候選 run 的 real test div_rel（單一樣本與 $K=8$ 平均）與真值中位數 | | |

閱讀「參考實作」中的兩個 repo（clone 到 repo 以外的暫存目錄，指定上表的 commit），回報：

- 「已知差異」1–8 是否屬實（附檔案:行號）；是否還有其他論文與程式碼的差異。
- 官方 PCFM 的 `pcfm_sample` 與本任務「背景」一節的演算法逐行對照，列出不一致之處。
- 官方 Jacobian 計算（`jacrev`、`chunk_size`）在本資料尺寸（$n = 20\times3\times64\times128 = 491{,}520$）下的預估記憶體。

另外回報：

- R6 與 R8 何者作為主要模型，列出兩者的成本與指標差異；**由我決定**（預設 R6）。
- 上表中任何與本任務「背景」一節不一致之處。

⛔ STOP

---

## Step 1　實作取樣器與約束介面（不跑任何模型推論）

新增 `fm/constrained.py`（不修改 `fm/paths.py` 的既有函式）：

1. **約束介面**：
   ```
   class Constraint:
       def h(self, y_norm, ctx) -> (B, m)   # y_norm: 正規化的 (B, T_out·C, H, W)；ctx: 每樣本的 sim 幾何等資訊
       linear: bool                         # True 時允許走 ECI 的封閉解路徑
   ```
   反正規化在 `h` 內部完成，使 autodiff 的 Jacobian 自動包含它。
2. **投影求解器** `gn_project(y, constraint, ctx, anchor=None, iters=1, eps=0.0)`：batched，計算 $JJ^\top+\epsilon I$（$m\times m$）並以 `torch.linalg.solve`（或 Cholesky）求解；$\epsilon$ 的預設值依 P9。$J$ 的計算方式（`vmap(jacrev)`，或以 $m$ 次 `vjp` 組出 $JJ^\top$）由你依記憶體量測決定，回報兩者的峰值記憶體與時間。
3. **取樣器** `constrained_sample(model, cond, shape, n_steps, constraint, ctx, mode, generator, newton_tol, newton_max_iter, n_mix=1, resample=None)`，`mode ∈ {"interp", "pcfm", "final_only", "eci"}`，對應 A1、A2、A3、A4。
   - `interp`：修正步驟為 identity。
   - `final_only`：中間步驟不投影，只做最終 Newton 投影。
   - `eci`：只在 `constraint.linear` 為 True 時允許；$n_\text{mix}$、$R$ 的語意與 ECI 官方實作 `models/functional.py: eci_sample` 相同（$R$ 以 ECI step 次數計數，`None` 表示不重抽）。
   - `relaxed`（預設 `None`，即關閉）：開啟時依「參考實作」第 5 點的形式，參數 `lam`、`step_size`、`num_steps`、`gamma_min` 預設與官方相同。
   - 每次呼叫回傳樣本，並可選擇回傳診斷資訊：每步投影前後的 $\|h\|$、最終 Newton 的迭代次數與每次迭代的殘差。
4. **接入**：`FMPredictor` 新增參數 `sampler`（預設 `"euler"`，行為不變）與 `constraint`；`sampler != "euler"` 時改呼叫 `constrained_sample`。噪聲的抽取方式須與 `euler_sample` 相同（同一 generator、同一形狀、同一順序）。

驗證（全部附輸出；只用合成 tensor 或少量真實資料的真值，不呼叫訓練好的模型）：

1. **向後相容**：`sampler="euler"` 時，固定 seed 下 `FMPredictor` 的輸出與修改前逐位元相同。
2. **線性約束等價**：以合成的指定數值約束（隨機 mask），比較 `gn_project` 一次的結果與封閉解（直接覆寫）的最大絕對誤差。
3. **投影的最小範數性質**：對非線性的測試約束，檢查 $u-a$ 是否落在 $J^\top$ 的值域內（以最小平方殘差表示），以及 $h(u)$ 在 Newton 迭代中的下降曲線。
4. **Jacobian 正確性**：以 `torch.autograd.gradcheck`（fp64、小尺寸）或有限差分檢查 $h$ 的 Jacobian。
5. **物理量一致性**：若 Step 2 使用 div_rel 類的約束，在 test 真值以外的 real 資料（train 或 val）上，確認 `h` 內部算出的 div_rel 與 `scripts/eval_xpred_width.div_rel` 的結果一致（列出最大差異）。
6. **Oracle 取樣**：以「永遠輸出某個固定場 $y^\*$ 的假網路」執行 `pcfm` 模式，確認最終輸出等於 $y^\*$ 投影到約束上的結果，且 $\|h\|<\varepsilon$。
7. **與官方實作對照**：在小尺寸的合成問題上（例如官方 `constraints.py` 的 heat IC + 質量守恆殘差，$n$ 為數千），以同一個假網路、同一組噪聲，比較本實作的單步 `pcfm` 更新與官方 `pcfm_sample` 的輸出（官方程式只在暫存目錄中執行，不進 repo），列出最大差異；官方最終投影因「已知差異」第 1 點無法直接呼叫，改以手動呼叫 `fast_project_batched(xi, h, max_iter=k)` 對照。
8. 列出所有新增與修改的檔案及 diff 摘要。

⛔ STOP

---

## Step 2　約束候選與目標值統計（只讀資料，不呼叫模型）

在 **real train 與 real val**（不得使用 test）上計算並回報：

- 每幀 div_rel 的分佈（中位數、四分位距、逐 sim 的中位數），定義依 D32；並列出 test 真值的中位數（RESULTS §4.2 的 0.681），**僅供對照，不用於設定目標值**。
- 障礙物與 PIV 無效格點的數值統計（依 Step 0 的發現）。

接著列出以下候選約束的可行性、約束數 $m$、是否線性、預估投影成本，**由我選擇**要施加哪些：

| 代號 | 約束 | 形式 | 備註 |
|---|---|---|---|
| C1 | 障礙物 / 無效格點的指定數值 | 線性，$m$ = 格點數 × 幀數 × 通道數 | 只在 Step 0 確認這些格點在資料中有確定的值時才成立 |
| C2 | 每幀 div_rel 等於目標值 $\tau$ | 非線性，$m = T_\text{out} = 20$ | $\tau$ 只能由 train / val 統計決定；`dit_xpred_width` 中 x-pred 單一樣本的 div_rel 已接近真值（0.677–0.679 vs 0.681），回報時一併列出這個資訊 |
| C3 | 稀疏觀測點的指定數值（例如 test 真值上的少數探針位置） | 線性 | **改變了任務設定**：推論時使用目標幀的部分真值作為輸入，屬於資料同化 / sim-to-real 校正的設定，不能和 A0 或 baseline 的數字直接比較；只有我明確同意才做，且必須在所有輸出中標示 |

其他約束（例如逐點零散度、不等式約束）不在本任務範圍內（P6；PCFM 只支援等式約束）。若你認為有其他可行的約束，列出來，不要自行加入。

⛔ STOP — 回報後由我決定約束組合與目標值。

---

## Step 3　Smoke test（僅此步允許短跑）

- 資料：real **val** 子集，前 128 筆；Euler $N=20$、$K=1$、noise seed 0；只用 Step 0 決定的主要模型。
- 對 A0、A1、A2、A3（以及 Step 2 選定的約束含線性約束時的 A4）各跑一次。

回報：

| 代號 | 每 batch 時間 | 峰值 GPU 記憶體 | 最終 $\|h\|$（中位數 / 最大值） | 最終 Newton 迭代次數（中位數 / 最大值） | 是否有不收斂或 NaN | val Rel L2（單一樣本，normalized 與物理單位） |
|---|---|---|---|---|---|---|

另外回報：

- A0 與 A1 的輸出差異（Rel L2、逐像素最大差），用來說明「只換取樣器、不加約束」本身的影響。
- 每步投影前後的 $\|h\|$ 曲線（A2）。
- 推估 Step 4 每個取樣方式所需的 GPU-h。

⛔ STOP — **等待我明確同意後才能進 Step 4。**

---

## Step 4　正式評估（需事先同意）

- 評估協定依 P7。實作上新增 `scripts/eval_pcfm_xpred.py`，或在 `eval_xpred_width.py` 以參數開關加入取樣器選擇（預設行為不變），兩者擇一並先回報方案。
- **A0 的數字**：先在 128 筆 test 上確認新腳本以 `sampler="euler"` 重現 `dit_xpred_width` S6 的 JSON（逐位元或列出差異來源），確認後可直接沿用 S6 的完整結果，不重跑。
- 對 A1、A2、A3（及 A4）跑 $N_{ar}=1$、$K=8$，報：
  - 官方 13 項指標（單一樣本與 $K=8$ 平均）、CRPS、spread
  - div_rel（單一樣本、$K=8$ 平均、真值）
  - 約束殘差 $\|h\|$ 的分佈（單一樣本與 $K=8$ 平均分別計算；注意 $K=8$ 平均是否仍滿足約束取決於約束是否線性，照實回報）
  - 徑向動能譜（第一個 test batch，與 S6 相同）
  - 推論時間：一次呼叫（20 幀）的時間與每幀時間，batch 1 與 batch 64，fp32，CUDA events
- $N_{ar}=10$ rollout **不在本步範圍內**；Step 4 回報後由我決定是否另外執行。

輸出：

- `S4_eval-test/summary/pcfm_xpred.csv`（每個取樣方式一列）
- 圖 1：各取樣方式的 Rel L2 與 div_rel 並列
- 圖 2：約束殘差的分佈（log scale）
- 圖 3：徑向動能譜，A0–A3 疊圖
- 圖 4：準確度 vs 每幀推論時間，放入 A0–A3 與既有 baseline 的點（baseline 計時沿用 S6 bench 結果）

⛔ STOP

---

## 每步回報格式

```
### Step N 回報
- 完成項目：
- 修改的檔案：
- 驗證結果（附數字 / 輸出）：
- 發現的不一致或疑點：
- 需要我決定的事項：
- 本步新增至 docs/pcfm_xpred/DECISIONS.md 的條目：
```
