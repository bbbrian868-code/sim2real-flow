# Task: Leray 投影（散度為 0）作為推論期硬約束的對照組

> 本檔是給 Claude Code 的執行清單。
> 接續 `docs/pcfm_xpred/`（P1–P31）。先前的約束只有 C2：每幀 div_rel 等於該樣本輸入最後一幀的 div_rel（τ_last，P19）。本任務加入「散度為 0」的對照，並量化這條約束在 sim 與 real 上各自的代價。
> 本任務**不訓練任何模型**，只對既有 checkpoint 做推論，以及對資料本身做投影。

---

## 目標

1. 實作 Leray 投影（逐點線性約束 $D\mathbf{u}=0$），接到既有的 x-pred 取樣器（`fm/constrained.py`）與評估腳本（`scripts/eval_xpred_width.py`）。
2. **真值 oracle**：把 Leray 投影直接套在 real test 與 sim test 的**真值**上，量出「即使模型完全正確，強制散度為 0 也會造成多少誤差」。這是散度為 0 這組在指標上的下限，不需要跑模型。
3. 在 R6（real 訓練）與 R6num（sim 訓練）上，比較以下取樣方式在 real test 上的表現，$N_{ar}=1$ 與 $N_{ar}=10$ 都跑。

| 代號 | 約束 | 投影時機 | 狀態 |
|---|---|---|---|
| A0 | 無 | — | R6：已有（S6、S8 的 A0）；R6num：$N_{ar}=1$ 已有（§7），$N_{ar}=10$ 待跑 |
| A2' | C2（τ_last） | 每步投影，反推噪聲退回 | R6：已有（S7、S8）；R6num：待跑 |
| A3' | C2（τ_last） | 只在最後投影（= Euler + 事後投影） | R6：已有（S7、S8）；R6num：待跑 |
| L-raw-free-final | Leray，存檔尺度，Ω 內約束，無壁面條件 | 只在最後 | 新 |
| L-raw-free-step | 同上 | 每步，反推噪聲退回 | 新 |
| L-raw-wall-final | Leray，存檔尺度，圓柱內設 0、約束區擴大到圓柱周圍 | 只在最後 | 新 |
| L-raw-wall-step | 同上 | 每步 | 新 |
| L-s-free-final | Leray，v/u 尺度修正 | 只在最後 | 新 |
| L-s-free-step | 同上 | 每步 | 新 |
| L-s-wall-final | Leray，v/u 尺度修正，壁面條件 | 只在最後 | 新 |
| L-s-wall-step | 同上 | 每步 | 新 |
| P-raw | 沿 Leray 方向的部分投影，使每幀 div_rel = τ_last | 只在最後 | 新 |

---

## 背景

### Leray 投影是 PCFM 投影在線性約束下的封閉解

把離散散度寫成線性算子 $D$（約束區內每個格點一列），約束 $h(\mathbf{u})=D\mathbf{u}=0$，PCFM 的投影公式變成

$$
\mathbf{u}_\text{proj} = \mathbf{u} - D^\top (DD^\top)^{-1} D\mathbf{u}
$$

$DD^\top$ 是離散 Laplacian（半正定），用矩陣自由的共軛梯度法（CG）求解 $(DD^\top)\phi = D\mathbf{u}$，再令 $\mathbf{u}\leftarrow\mathbf{u}-D^\top\phi$。一次即精確，不需 Newton 迭代。$D^\top$ 用 `torch.func.vjp` 取得（$D$ 線性，vjp 就是精確的伴隨）。

**不用 C2 設 τ = 0 的理由**：div_rel 的分子是範數，在 0 處不可微，PCFM Theorem A.1 的假設（$h\in C^2$、Jacobian full rank）在解處失效；對純量範數做 Gauss–Newton 等於沿單一方向的最速下降，收斂慢且不是最小改動解。

### 參考實作（已在隨機場上驗證）

以下程式已在 CPU、64×128、含邊界與圓形障礙物的隨機場上測過：散度從 1.6e3 降到 1.6e-7（fp64，tol 1e-10，190 次 CG 迭代），冪等性與正交性皆通過。可作為起點，但必須依本任務的定義（尺度 $s$、壁面條件、逐樣本幾何、batch 維度）改寫並重新驗證。

```python
def div_central(u, v, om, dx, dy, s=1.0):
    """與 D32 的 div_rel 同一個離散散度（中央差分、邊緣行列為 0）；om: 約束區的 float mask。
    s: v/u 尺度修正（D_s 用 1/r，D_raw 用 1）。"""
    a = F.pad((u[..., :, 2:] - u[..., :, :-2]) / (2 * dx), (1, 1, 0, 0))
    b = F.pad((v[..., 2:, :] - v[..., :-2, :]) / (2 * dy), (0, 0, 1, 1))
    return (a + s * b) * om

def leray_cg(u, v, om, dx, dy, s=1.0, tol=1e-10, max_iter=5000):
    D = lambda u_, v_: div_central(u_, v_, om, dx, dy, s)
    _, Dt = torch.func.vjp(D, u, v)              # D 線性 → vjp = Dᵀ
    A = lambda p: D(*Dt(p))                      # A = D Dᵀ
    dims = (-2, -1)
    b = D(u, v); x = torch.zeros_like(b); r = b.clone(); p = r.clone()
    rs = (r * r).sum(dims, keepdim=True); bnorm = rs.sqrt().clamp_min(1e-30)
    for it in range(max_iter):
        Ap = A(p)
        alpha = rs / (p * Ap).sum(dims, keepdim=True).clamp_min(1e-30)
        x = x + alpha * p; r = r - alpha * Ap
        rs_new = (r * r).sum(dims, keepdim=True)
        if (rs_new.sqrt() / bnorm).max() < tol:
            break
        p = r + (rs_new / rs) * p; rs = rs_new
    du, dv = Dt(x)
    return u - du, v - dv, it + 1
```

方法 A（週期邊界 + FFT，$\hat{\mathbf{u}} \leftarrow \hat{\mathbf{u}} - \tilde{\mathbf{k}}(\tilde{\mathbf{k}}\cdot\hat{\mathbf{u}})/|\tilde{\mathbf{k}}|^2$，修正波數 $\tilde k_x=\sin(k_x\Delta x)/\Delta x$）**只用於驗證**方法 B，不作為實驗組（cylinder 非週期、有障礙物）。

### 部分投影

令 $\mathbf{u}_\alpha=\mathbf{u}-\alpha D^\top\phi$（$\phi$ 為完全投影的解，同一個 $D$）。因為 $D\mathbf{u}_\alpha=(1-\alpha)D\mathbf{u}$，散度精確縮放 $(1-\alpha)$ 倍。對每一幀以二分法求 $\alpha$ 使 div_rel（D32 定義）$=\tau_\text{last}$。這只是沿單一方向的啟發式，**不是**滿足 div_rel $=\tau$ 的最小改動解（後者即 C2 的 A3'），兩者的比較正是此組的目的。

### 已知會影響數字的事實

- **real 的 v/u 尺度**：`docs/realdata/REPORT_2026-10-29_real-data-checks.md` §4 與 `REPORT_2026-10-29_vu-scale-and-rerank.md` §2：real 的 v 相對 u 大約 2.8 倍；每條 trajectory 的 CLFT β/α 中位數 2.832（2.823–2.854）。是 v 太大還是 u 太小無法判定，但散度約束只取決於比例。
- **圓柱內部的值**：real 資料圓柱內部（約 2.4% 格點）的 u 平均 0.48 U∞（同報告 §2b），官方指標在全網格計算，所以壁面條件版本把圓柱內設 0 會直接在這些格點上計入誤差。
- **評估迴圈的正規化往返**（`docs/pcfm_xpred/RESULTS` §5）：被評分的 u、v 分別乘上約 0.999986、1.000105。投影後散度為 0 的場，評估重算時 div_rel 預期約在 1e-4 量級而非 0；這不是投影失敗。
- **反推噪聲退回**（P30）：`--retreat implied` 不加約束時代數上等於 Euler；「只在最後投影」= Euler 樣本 + 事後投影一次。

---

## 全域規則（每一步都適用）

1. **每完成一個 Step 就停下回報**，等我回覆「繼續」才進下一步。
2. **未經明確同意，不得啟動任何正式推論**（smoke test 僅限 Step 3 規定的範圍；Step 2 的真值 oracle 只處理資料、不呼叫模型，但仍須在該步內完成並停下回報）。本任務不得啟動任何訓練。
3. **不得修改既有預設行為**：新功能以新檔案或參數開關加入；`fm/constrained.py`、`scripts/eval_xpred_width.py` 的既有呼叫（含 `--sampler`、`--retreat` 的所有既有組合）結果必須不變。
4. **遇到文件或程式碼語意不明時，不得自行假設**，列出選項並回報。
5. **發現架構或設定前後不一致時，直接標出來**，不要自行做局部修補。
6. **不得使用 test split 決定任何東西**：checkpoint、超參數、尺度 $s$、容忍值都只能由 train / val 或既有文件決定。真值 oracle 在 test 上計算屬於報告，不得據以調整任何設定。
7. 只回報數字與觀察到的現象，不要替結果下研究結論。
8. **維護決策紀錄** `docs/leray_xpred/DECISIONS.md`（格式同 `docs/pcfm_xpred/DECISIONS.md`：編號、日期、Step、決定、理由；理由欄空白表示使用者偏好；推翻時新增一筆並在決定欄註明取代的編號）。Step 0 開始前建立此檔，寫入下方 L1–L15；理由必須是我給的理由，不得自行代填；每步回報的最後附上本步新增的決策編號。
9. 本任務檔原樣存為 `docs/leray_xpred/TASK_2026-11-22_leray-xpred.md`；結果根目錄 `/work/b314513067/pi-lfm/results/fm/2026-11-22_leray-xpred/`，子目錄命名沿用 `S{step}_{描述}/`；README 的實驗表新增一列。

---

## 已確定的決策

| # | 決定 | 理由 |
|---|---|---|
| L1 | 新增「散度為 0」對照：以逐點線性約束 $D\mathbf{u}=0$ 的 Leray 投影（方法 B：遮罩、非週期、矩陣自由 CG）實作；不以 C2 設 τ = 0 | div_rel 的範數在 0 處不可微，Gauss–Newton 的收斂假設失效；Leray 投影是線性約束的精確最小改動解 |
| L2 | 方法 A（FFT）只用於驗證方法 B，不作為實驗組 | cylinder 非週期且有障礙物，FFT 會讓邊界互相繞接 |
| L3 | 只跑反推噪聲退回（`retreat=implied`），不再跑初始噪聲退回 | P30–P31 已顯示初始噪聲退回本身就讓誤差變大；反推版本不加約束時等於 Euler，約束的效果才能被分離出來 |
| L4 | Leray 的實驗組為 {存檔尺度 D_raw, 尺度修正 D_s} × {無壁面條件 free, 壁面條件 wall} × {只在最後, 每步} 共 8 組；另加部分投影 P-raw 1 組（沿 D_raw、free 的 Leray 方向，只在最後，目標 τ_last） |  |
| L5 | 加入真值 oracle：對 real test 與 sim test 的真值做 Leray 投影，報告投影前後的指標差與 div_rel；real 做 4 種（D_raw / D_s × free / wall），sim 做 2 種（D_raw × free / wall） | 量出「約束本身的代價」，與模型無關；sim 與 real 並列可量化同一條約束在兩種資料上的代價差 |
| L6 | D_s 的尺度：$s=1/r$，$r$ 為每條 trajectory 的 CLFT β/α（`vu-scale-and-rerank` 報告 §2 的 v/r 變體所用的值）；sim 一律 $s=1$ | 三種獨立估計都指向 v/u 約大 2.8 倍；用與既有「v/r」評估變體相同的逐 trajectory 值，數字可以和既有表格對照 |
| L7 | 壁面條件版本（wall）：圓柱內部格點的 u、v 設為 0 且不再被投影修改；散度約束區從 Ω 擴大為 Ω_wall（Ω 加上圓柱周圍、因膨脹而被排除的流體格點，確切定義依 Step 0 的盤點，由我確認）；PIV 無效格點仍排除 | 只覆寫圓柱內部而不擴大約束區，差分不會碰到圓柱格點，條件對 Ω 內的解沒有作用 |
| L8 | 所有組別的指標同時報：官方（全網格）、排除圓柱內部、strict Ω；以及既有的 v/r 評估變體 | 壁面條件會在圓柱內部（真值 0.48 U∞）計入誤差；v/u 尺度問題讓 v 的權重偏大 |
| L9 | 模型：R6（real 訓練，`best.pt` step 50000）與 R6num（sim 訓練，`best.pt` step 4000，以 real val 選點，屬上限，P28）都跑，不指定主模型 |  |
| L10 | $N_{ar}=1$（K = 8）與 $N_{ar}=10$（K = 1）都跑；test noise seed 1234、val seed 0（同 P12）；C2 組在 $N_{ar}=10$ 時每輪的目標值沿用真實輸入最後一幀（同 P25）；Leray 組不需要目標值 |  |
| L11 | 投影在**存檔的物理單位**下做（反正規化 → 投影 → 再正規化，統計量同 `DivRelConstraint`：`mean_targets` / `std_targets`）；壓力通道不動；CG 以 fp64 計算，相對容忍值預設 1e-10、上限 5000 次，Step 3 量測後再確認 | 正規化空間的散度沒有物理意義；fp64 的額外成本相對於一次 DiT forward 可忽略 |
| L12 | 評估協定同 P7、P20–P22：real test 4827 筆（$N_{ar}=1$）/ 4672 筆（$N_{ar}=10$），Euler 20 步，fp32，EMA，官方評估迴圈；C2 組的最終 Newton 投影設定同 P17、P22 | 與 `pcfm_xpred` 的所有數字可直接並列 |
| L13 | R6 已有的 A0、A2'、A3'（S6、S7、S8）不重跑：先在前 128 筆 test 上確認新程式以相同參數重現（$N_{ar}=1$ 逐位元或列出差異來源；$N_{ar}=10$ 相對差 ≤ 1e-3，同 P31），再沿用既有結果 |  |
| L14 | R6num 不改用 sim val 重新選 checkpoint，維持 L9 的 `best.pt`（step 4000），所有 R6num 的數字標註為上限（P28） |  |
| L15 | 不做 R6num 在 sim test 上的模型評估；sim test 只用於 Step 2 的真值 oracle |  |

---

## Step 0　盤點（唯讀，不改任何檔案）

| 項目 | 值 | 出處（檔案:行號） |
|---|---|---|
| R6、R6num 的 checkpoint 路徑與步數 | | |
| D32 的 Ω 如何產生（`run_E.geometry`）：邊界幾格、障礙物膨脹幾格、PIV 無效格點的判定；圓柱本體（未膨脹）的 mask 能否單獨取得 | | |
| Ω_wall 的候選定義（L7）：Ω 加上「膨脹環」中的流體格點；列出每條 real trajectory 的格點數（Ω、膨脹環、圓柱本體、PIV 無效） | | |
| 中央差分在 Ω_wall 的邊緣會讀到哪些格點；確認圓柱本體格點會進入差分（否則壁面條件無作用，回報） | | |
| real 資料圓柱本體格點的 u、v 數值分佈（物理單位）；sim 資料圓柱本體格點的數值 | | |
| 每條 real trajectory 的 β/α（L6）的來源檔與數值；與 `vu-scale-and-rerank` 報告 v/r 變體所用的值是否逐位元相同 | | |
| sim test 的取得方式：官方 loader 提供的解析度（64×128 或 128×256）、格距、座標方向與 v 的符號慣例、是否帶 10% 乘性噪聲；sim 的 Ω 與圓柱 mask 是否已有（CLFT 或本 repo） | | |
| `fm/constrained.py` 可接入線性約束的位置：`Constraint.linear` 目前的用法；`gn_project` 與 `constrained_sample` 要如何分派到封閉解投影 | | |
| `scripts/eval_xpred_width.py` 要新增的參數（約束種類、尺度、壁面條件）與輸出檔名規則；既有 `-implied` tag 的處理 | | |
| 既有的區域指標變體（strict Ω、v/r、排除圓柱）在哪支程式（`scripts/realdata_checks.py`？）、能否在 eval 內直接重用 | | |
| R6num 的 $N_{ar}=10$：A0、A2'、A3' 是否真的都沒跑過 | | |

另外回報任何與本任務「背景」不一致之處。

⛔ STOP

---

## Step 1　實作與驗證（不呼叫任何訓練好的模型）

新增 `fm/leray.py`（或放進 `fm/constrained.py`，先回報方案）：

1. `leray_project(u, v, region, dx, dy, s=1.0, fixed=None, tol, max_iter, return_info)`：方法 B。
   - `region`：約束區 float mask（free = Ω，wall = Ω_wall）；`fixed`：不允許修改的格點 mask（wall = 圓柱本體）。wall 版本先把 `fixed` 格點的 u、v 設為 0，並把 $D$ 定義在「`fixed` 以外的變數」上（例如 $D(u\cdot f, v\cdot f)$，$f$ = 1 − fixed），使 $D^\top\phi$ 在 `fixed` 格點上為 0。
   - 逐樣本的幾何（region、dx、dy、s）以 batch 維度傳入；dx、dy 的正負號與 D32 相同（來自座標陣列）。
   - info：每幀 CG 迭代次數、最終相對殘差、投影前後 region 內 $\max|D\mathbf{u}|$、修改量 $\|\Delta\|$ 在 region 內 / region 外 / fixed 格點上的分佈。
2. `leray_fft(u, v, dx, dy, s=1.0, match_central_diff=True)`：方法 A，只供驗證。
3. 線性約束類別 `LerayConstraint`（`linear = True`，提供 `project(y_norm, ctx)`：反正規化 → `leray_project` → 再正規化，壓力通道原樣保留）；部分投影類別 `PartialLerayConstraint`（`project` 以二分法求每幀的 α）。
4. `constrained_sample`：當 `constraint.linear` 為 True（或提供 `project`）時，每步投影與最終投影改呼叫 `constraint.project`，不走 `gn_project`；`retreat="implied"` 的退回公式不變（反推噪聲用投影前的 x_raw）。
5. `eval_xpred_width.py`：以新參數選擇約束（例如 `--constraint c2|leray|partial`、`--div-scale raw|vr`、`--wall`），預設值重現既有行為；輸出檔名加上可區分 12 組的 tag。評估時另外輸出 L8 的區域變體指標，以及 div_rel 的兩種尺度（存檔、v/r 修正）。

驗證（全部附輸出）：

1. **向後相容**：既有所有 `--sampler` × `--retreat` 組合，在固定 seed 下與修改前逐位元相同。
2. **與顯式矩陣比對**：在小網格（例如 8×12，含遮罩與一個 fixed 區塊）上顯式建出 $D$，以 `torch.linalg.pinv` 計算 $\mathbf{u}-D^\top(DD^\top)^+D\mathbf{u}$，與 `leray_project` 的最大差（free / wall，s = 1 與 s ≠ 1）。
3. **投影性質**：region 內 $\max|D\mathbf{u}_\text{proj}|$；冪等性（再投影一次的最大差）；正交性 $\langle\mathbf{u}-\mathbf{u}_\text{proj},\mathbf{u}_\text{proj}\rangle$（wall 版本在可變格點上計算）；wall 版本 fixed 格點的值恰為 0。
4. **方法 A 對照**：把 $D$ 暫時換成週期版（`torch.roll` 差分、region 全為 1），`leray_project` 與 `leray_fft(match_central_diff=True)` 的最大差。
5. **部分投影**：在 real val 的 64 筆真值上，對 α ∈ [−1, 1] 取 41 點畫出每幀 div_rel(α)，回報是否單調；二分法後 div_rel 與 τ 的最大差；無解（τ 落在可達範圍外）的幀數與處理方式（先回報，不自行決定）。
6. **與 D32 一致**：在 real val 真值上，`LerayConstraint` 內部算出的 div_rel（s = 1、region = Ω）與 `scripts/eval_xpred_width.div_rel` 的最大差。
7. **正規化往返的預期值**：取 real val 真值投影後的場，送進官方評估迴圈的反正規化 / 正規化往返，回報評估重算的 div_rel 與 $\max|D\mathbf{u}|$，作為 Step 4 的預期量級。
8. 列出所有新增與修改的檔案及 diff 摘要。

⛔ STOP

---

## Step 2　真值 oracle（只處理資料，不呼叫模型）

對 test 真值本身做 Leray 投影（每個輸出幀獨立投影），與原始真值比較。$N_{ar}=1$ 的切窗（4827 筆，每筆 20 幀）。

| 資料 | 變體 |
|---|---|
| real test | D_raw-free、D_raw-wall、D_s-free、D_s-wall |
| sim test | D_raw-free、D_raw-wall（s = 1） |

每個變體回報：

- 投影後場相對原始真值的 Rel L2（官方定義，u、v），以及 L8 的區域變體（全網格、排除圓柱、strict Ω、v/r）
- 投影前後的 div_rel 中位數（兩種尺度都報）
- 修改量在 region 內、region 外、圓柱本體上的佔比
- 動能（KE）的相對變化
- CG 迭代次數分佈

並另外列出：同一組指標下，R6 A0 與 R6num A0 的誤差（既有數字），方便判讀「約束的代價」相對於「模型本身的誤差」有多大。

sim test 若在 Step 0 判定無法取得，只做 real，並回報原因。

⛔ STOP

---

## Step 3　Smoke test（僅此步允許短跑）

- 資料：real **val** 前 128 筆；Euler 20 步、K = 1、noise seed 0；R6 與 R6num 各跑 8 個 Leray 組與 P-raw；R6num 另跑 A0、A2'、A3'。
- 回報：

| 模型 | 組別 | 每批 64 筆時間 | 峰值 GPU 記憶體 | CG 迭代次數（中位數 / 最大） | region 內 max\|Du\| | 評估重算 div_rel 中位數 | val Rel L2（官方、排除圓柱） | NaN / 未收斂 |
|---|---|---|---|---|---|---|---|---|

- 另外回報：每步投影組（step）中，每步投影前的 $\max|D\mathbf{u}|$ 曲線；確認「只在最後」組的結果等於「A0 的 Euler 樣本 + 事後投影」（逐位元或列出差異）。
- 推估 Step 4（$N_{ar}=1$）與 Step 5（$N_{ar}=10$）每一組所需的 GPU-h 與總量。

⛔ STOP — **等待我明確同意後才能進 Step 4。**

---

## Step 4　正式評估 $N_{ar}=1$（需事先同意）

- 先做 L13 的重現檢查（R6 的 A0、A2'、A3'，前 128 筆 test）。
- R6：8 個 Leray 組 + P-raw。R6num：A2'、A3'（A0 沿用 §7）+ 8 個 Leray 組 + P-raw。
- 每組報：官方 13 項指標（單一樣本與 K = 8 平均）、CRPS、spread、L8 的區域變體、div_rel（兩種尺度；單一樣本、K = 8 平均、真值）、約束殘差（Leray 組：評估重算的 $\max|D\mathbf{u}|$ 與 div_rel；C2 組：同 `pcfm_xpred`）、徑向能譜（第一個 test batch）、推論時間（batch 1 與 64）。
- 輸出：`S4_eval-nar1/summary/leray_xpred_nar1.csv`（每個模型 × 組別一列，含 A0、A2'、A3' 的既有數字）；圖：
  1. 每個模型一張：各組的 Rel L2（官方與排除圓柱）與 div_rel 並列，並畫出 Step 2 的真值 oracle 作為水平參考線
  2. 約束殘差分佈（log scale）
  3. 能譜疊圖
  4. 準確度 vs 每幀推論時間

⛔ STOP

---

## Step 5　正式評估 $N_{ar}=10$（需事先同意）

- 先在前 128 筆 test 上確認新程式不加約束時重現 A0 的 10 輪結果（相對差 ≤ 1e-3，同 P31）。
- R6：8 個 Leray 組 + P-raw（A0、A2'、A3' 沿用 S8）。R6num：A0、A2'、A3' + 8 個 Leray 組 + P-raw。
- 投影後的幀照官方迴圈回饋為下一輪輸入（不另外處理）。P-raw 與 C2 組每輪的目標值沿用真實輸入最後一幀（L10）。
- 每組報：200 幀整體 Rel L2、每一輪的 Rel L2（官方與排除圓柱）、每一輪的 div_rel 中位數（兩種尺度）與約束殘差、CG 迭代次數。
- 輸出：`S5_eval-nar10/summary/leray_xpred_nar10.csv`；圖：每個模型一張每輪 Rel L2 曲線、一張每輪 div_rel 曲線。

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
- 本步新增至 docs/leray_xpred/DECISIONS.md 的條目：
```
