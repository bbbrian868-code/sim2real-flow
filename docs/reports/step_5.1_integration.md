# Step 5.1 報告：整合方案

**問題與答案**（行號對應官方 `realpdebench/`）
- **官方訓練迴圈能否接受自訂 loss 或 training step？** 只能透過 `model.train_loss(input, target)` 這個介面（`train.py:328`），模型只收到 input 和 target。optimizer 寫死為 Adam（`train.py:290`），scheduler 只有 step 和 cosine 兩種（`:291-296`）。**沒有** EMA、warmup、bf16、DDP，也沒有 AdamW 或 gradient clipping 以外的設定（clip 在 `:330`）。
  - FM 的 loss 本身可以塞進 `train_loss`（y0 和 t 在裡面抽）。但驗證用的是 `model(input)`，對 FM 來說就是單次採樣（`:361`），而且沒有 EMA；規格 5.2 要求的訓練設定（AdamW、warmup+cosine、EMA、bf16、DDP）官方迴圈全部做不到。
- **加噪和模態遮蔽在哪裡做？** 都在 dataset 的 `__getitem__` 裡做（`data/fluid_hf_dataset.py:297-327`），不在訓練迴圈。所以直接 import 官方 dataset 就能完整沿用。
- **指標在哪裡計算？模型要提供什麼介面？** 指標由 `utils/metrics.py:24` 的 `eval_metrics(pred, target, c, batch_size)` 計算，輸入是反正規化後的 `(B, T, H, W, C)`。模型只要提供 `forward(x_normalized) -> y_normalized`，形狀 `(B, T_out, H, W, C)`（`eval.py:315`）。另外需要 `load_checkpoint` 和 `train_loss`（`model/model.py`）。模型註冊在 `model/load_model.py`，這是官方原始碼。

**方案**
- **(A) 註冊成官方模型**：要在 `load_model.py` 加分支，會**修改官方原始碼（違反規則 4）**。就算用 monkeypatch 繞過，還是拿不到 EMA、DDP、bf16、AdamW 和 warmup。
- **(B) 自寫訓練迴圈**：import 官方的 `CylinderHFDataset`（包含加噪和遮蔽）、`GaussianNormalizer`（`mean_std.pt`）、`eval_metrics`。評估使用 `pilfm/official_eval.py`，它逐行複製 `eval.py:290-352` 的迴圈，Step 3.1 控制組已驗證和官方一致。

**選擇：(B)**（D-009，附錄 B 第 5 項）。

**附帶的實作**：`pilfm/fast_dataset.py` 是官方 dataset 的子類別。官方每取一筆樣本就要解碼整條軌跡（numerical 每個欄位約 0.5 GB），batch 64 時每步要好幾秒，這是瓶頸。子類別改成直接從 memory-mapped Arrow buffer 切片，其餘程式碼逐行照搬，包括 `random.random()` 遮蔽和 `torch.randn_like` 噪聲的呼叫順序。測試 (f) 驗證：在相同 RNG 狀態下，它和官方類別的輸出逐位元相同（D-015）。
