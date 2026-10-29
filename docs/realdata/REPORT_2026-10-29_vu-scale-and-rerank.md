# v/u 尺度查證與修正評估下的模型排名（2026-10-29）

接續 [REPORT_2026-10-29_real-data-checks.md](REPORT_2026-10-29_real-data-checks.md)。結果檔：`pi-lfm/results/realdata/2026-10-29_vu-scale-and-rerank/`（rerank_summary.json、h5_inspect.json、h5_coords/coords.json）。
程式：`pi-lfm-code/scripts/realdata_checks.py`（模式 `h5`、`rerank`、`rerank-summary`）。
Job：419316（normal，28 分鐘，Transolver 在 batch 64 時 OOM，已完成的模型保留）、419353（normal，13 分鐘，補完其餘模型）、419354、419373（dev，各數秒）。合計約 0.7 GPU-h。

## 1. v/u 尺度：公開資料能不能查證

### 1a. 公開發布的內容
- 下載站 `realpdebench.westlake.edu.cn` 的 `cylinder_h5.tar`（204.5 GB）：以 HTTP range 逐一讀 tar 標頭列出內容（未整包下載；清單在 `downloads/2026-10-29_realpdebench-h5-sample/tar_listing_cylinder_h5.txt`）。只有 `real/` 92 個 h5、`numerical/` 92 個 h5、10 個索引 `.pt` 和 `MANIFEST.txt`。
- GitHub repo（`AI4Science-WestlakeU/RealPDEBench`）：沒有 PIV 原始向量、粒子影像或標定檔；只有模擬用的 `.pde` 腳本。
- 論文（arXiv 2601.01829 HTML 版）寫「released a subset of the raw particle image sequences and calibration files in the database … the raw velocity vector field data … has also been made available」，但上述兩處都找不到。

### 1b. 一個 real h5（Re 1781，654 MB，自 tar 以 byte range 取出）

| 項目 | 結果 |
|---|---|
| 結構 | `measured_data/u`、`measured_data/v`（3990×64×128 float32）、`vo`、`x`、`y`（3990×64×128 float32）、`t`、`re`；沒有任何 attribute |
| u、v、t 與 HF Arrow 版 | 逐位元相同 |
| x、y | 每幀都相同，等於 Arrow 版的網格；dx = 2.538 mm、dy = 2.731 mm（均勻，dx / dy = 0.929） |
| t | 0.050 s 到 19.995 s，間隔 5 ms，共 3990 幀 |

- t 從 0.050 s 開始，等於少了前 10 幀（50 ms），和 10 幀滑動平均的窗口長度相同。
- h5 沒有 Arrow 版以外的資訊，無法用來判定 v/u 尺度。

### 1c. 與論文敘述的出入
- 論文說自由流區的 PIV 速度有和水槽預先標定的流速比對。
- 存檔的上游自由流 u / U_inf（U_inf = Re·ν / D，ν = 1e-6）隨 Re 從 1.97 降到 0.75，CLFT 擬合為 u ≈ 0.53·U_inf + 0.09 m/s。帶截距的線性關係無法用單一常數（例如 ν 的溫度修正）解釋。

## 2. 修正評估下的排名（real test，4827 個樣本，N_ar = 1）

設定：
- 模型：
  - Phase 4 的 U-Net、Transolver（real、finetune 各 3 個 seed）、DeepONet（各 1 個 seed）；numerical（只用 sim 訓練）各 1 個 seed。
  - R6 x-pred（best.pt，Euler 20，K = 8，雜訊順序與 Step 6 相同）。
  - persistence（最後一個輸入幀）。
- 指標：全部是 Rel L2（每個樣本的 ‖pred − target‖ / ‖target‖ 取平均）。
- 變體：
  - 區域：全部格點；或 strict Ω（排除 Ω 外和假影區）。
  - 輸出幀：1–20；或 11–20。
  - v/r：pred 和 target 的 v 都除以該 trajectory 的 CLFT β/α（中位數 2.832，範圍 2.823–2.854）。只改變 u、v 的相對權重；因為 Rel L2 不受整體尺度影響，「v 除以 r」和「u 乘以 r」的結果相同。
- 自我檢查：official 欄與 Phase 4、Step 6 相同（U-Net real 平均 0.0620、R6 0.0785 / K-mean 0.0625、DeepONet 0.1429、Transolver real 平均 0.1449）。

| 模型（seed 數） | official | strict Ω | 幀 11–20 | v/r | strict + 11–20 | strict + 11–20 + v/r | 只算 u | 只算 v |
|---|---|---|---|---|---|---|---|---|
| U-Net real（3） | 0.0620 ± 0.0003 | 0.0613 | 0.0739 | 0.0370 | 0.0741 | 0.0403 ± 0.0003 | 0.0311 | 0.1908 |
| U-Net finetune（3） | 0.0618 ± 0.0002 | 0.0611 | 0.0737 | 0.0366 | 0.0739 | 0.0400 ± 0.0004 | 0.0306 | 0.1905 |
| R6 K = 8 平均 | 0.0625 | 0.0619 | 0.0760 | 0.0362 | 0.0765 | 0.0408 | 0.0298 | 0.1942 |
| R6 單一樣本 | 0.0785 | 0.0780 | 0.0957 | 0.0452 | 0.0964 | 0.0513 | 0.0371 | 0.2448 |
| Transolver finetune（3） | 0.1303 ± 0.0007 | 0.1331 | 0.1313 | 0.0711 | 0.1345 | 0.0706 ± 0.0004 | 0.0561 | 0.4173 |
| Transolver real（3） | 0.1449 ± 0.0072 | 0.1486 | 0.1454 | 0.0784 | 0.1492 | 0.0780 ± 0.0041 | 0.0613 | 0.4635 |
| DeepONet finetune（1） | 0.1345 | 0.1375 | 0.1354 | 0.0780 | 0.1390 | 0.0765 | 0.0646 | 0.4313 |
| DeepONet real（1） | 0.1429 | 0.1455 | 0.1440 | 0.0841 | 0.1471 | 0.0823 | 0.0703 | 0.4548 |
| persistence | 0.1807 | 0.1924 | 0.2303 | 0.0836 | 0.2459 | 0.1100 | 0.0537 | 0.5822 |
| U-Net numerical（1） | 0.2086 | 0.2163 | 0.2194 | 0.1771 | 0.2274 | 0.1944 | 0.1717 | 0.4541 |
| Transolver numerical（1） | 0.2992 | 0.3140 | 0.2995 | 0.1860 | 0.3142 | 0.1914 | 0.1597 | 0.8724 |
| DeepONet numerical（1） | 0.3206 | 0.3425 | 0.3212 | 0.1979 | 0.3436 | 0.2071 | 0.1697 | 0.9776 |

20 個單一模型的排名與 official 的 Kendall τ：

| 變體 | strict Ω | 幀 11–20 | v/r | strict + 11–20 | strict + 11–20 + v/r（r = 2.6 / 3.0） | 只算 u | 只算 v |
|---|---|---|---|---|---|---|---|
| τ | 1.00 | 0.96 | 0.86 | 0.95 | 0.94（0.96 / 0.93） | 0.76 | 0.93 |

觀察：
- **v 的比重：** v 占 target 能量的 9.8%，卻占訓練過的模型平方誤差的 73–81%（U-Net 0.73、R6 0.75、Transolver 0.80；persistence 0.91）。v/r 修正後，訓練過 real 的模型 Rel L2 降低 40–46%，persistence 降低 54%，numerical 模型降低 15–38%。
- **最前段：** U-Net 6 個 checkpoint 和 R6 K = 8 平均彼此相差，official 下 ≤ 0.001，strict + 11–20 下 ≤ 0.003。
  - official 下，R6 K-mean 比 U-Net real 平均高 0.0005。strict + 11–20 + v/r 下也高 0.0005，排第 7。
  - 只做 v/r 修正時，R6 K-mean 是 0.0362，與 U-Net 最好的 checkpoint 並列第一。
  - 只算 u 時，R6 K-mean（0.0298）低於全部 6 個 U-Net（0.0302–0.0316）；只算 v 時則高 0.003。
- **finetune 與 real：**
  - Transolver：0.1303 對 0.1449，修正後 0.0706 對 0.0780；real 的 seed 標準差 0.0072，finetune 0.0007。
  - DeepONet：0.1345 對 0.1429，修正後 0.0765 對 0.0823。
  - U-Net：差 0.0002–0.0004，與 seed 標準差相當。
- **只用 sim 訓練（zero-shot）：**
  - official 下，U-Net numerical（0.209）比 persistence（0.181）差；strict + 11–20 下比 persistence 好（0.227 對 0.246）；v/r 下又比 persistence 差很多（0.177 對 0.084）。
  - 它的誤差主要在 u：只算 u 時是 0.17，訓練過 real 的模型是 0.03–0.07。U-Net numerical 的 v 占平方誤差 0.39。
  - zero-shot 的輸入沒有修正已知的 sim/real 差異：v 符號相反、圓柱位置差 29 mm、u 的水準不同（real 約 0.53·U_inf + 0.09，sim 約 0.95·U_inf）。
- **strict Ω、幀 11–20 的 RMSE（m/s）：** U-Net 0.0145–0.0146、R6 K-mean 0.0150、R6 單一樣本 0.0188。前一份報告估計的雜訊下限是 0.0136–0.0153。

## 3. 限制
- v/r 修正假設 CLFT 的 β/α 就是 v 相對 u 的尺度誤差；第 1 節無法證實或排除這一點。
- DeepONet 和 numerical 模型都只有 1 個 seed。
- R6 只有一組雜訊 seed（1234）。
- zero-shot 的列只反映「直接把 real 輸入給 sim 模型」，不代表修正已知差異後的遷移能力。
