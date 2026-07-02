# Bug 修復報告

- **日期**: 2026-07-02 11:37
- **任務**: 修復 CSV 批次預測回傳非法 JSON（`NaN` token）導致前端解析失敗
- **範圍**: `src/web/ml_api.py`（批次預測端點序列化）

## 現象

於 `/ml/predict` 上傳含空 `Cabin` 欄的 CSV，前端顯示錯誤：

```
無法連線到伺服器：Unexpected token 'N', ..."Cabin": NaN, ... is not valid JSON
```

- HTTP 狀態實為 200，模型預測其實成功。
- 錯誤發生在瀏覽器 `response.json()`（`JSON.parse`）階段，非網路問題。

## 根因

`POST /api/ml/predict/batch` 直接把 pandas DataFrame 序列化：

```python
# 舊寫法（src/web/ml_api.py）
return jsonify(
    {"results": result.to_dict(orient="records"), "total": len(result)}
), 200
```

- CSV 空值被 pandas 讀成 `NaN`，`result` 仍保留原始輸入欄（含 `Cabin`）。
- Flask 預設 JSON provider 會把 `float('nan')` 輸出成 **`NaN`** token。
- `NaN` 不是合法 JSON，瀏覽器嚴格解析直接拒絕。

### 為何測試沒抓到

先前以 Python `test_client` + `r.get_json()` 驗證會「通過」，因為 Python `json.loads` 預設 `parse_constant` **容許** `NaN`／`Infinity`，比瀏覽器寬鬆，掩蓋了問題。

## 修法（DRY）

專案內 `_csv_to_records` 已有「`NaN → None`、numpy 純量 → 原生型別」的邊界轉換，正是 JSON 輸出所需。將其一般化為 `_frame_to_records`，同時服務於「CSV 輸入正規化」與「批次結果序列化」兩處：

```python
# 新寫法
return jsonify(
    {"results": _frame_to_records(result), "total": len(result)}
), 200
```

- 空 `Cabin` now 序列化為 `null`（合法 JSON）。
- 輸入端 `validate_passengers(_frame_to_records(frame))` 同步改名，維持單一路徑。

## 驗證

- [x] 以嚴格 JSON 解析（`parse_constant` 拋錯，等同瀏覽器）重打端點 → 通過，`Cabin = null`。
- [x] `uv run pytest tests/test_ml_api.py` → 30 passed。
- [x] 測試資料：`sample_batch.csv`（5 筆，含 3 筆空 Cabin）。

## 行動項目

- [x] 補一條**嚴格 JSON**斷言的回歸測試（拒絕 `NaN`／`Infinity` token），避免 `get_json()` 寬鬆解析再度掩蓋。
  → `tests/test_ml_api.py::test_batch_empty_cabin_serializes_as_valid_json_null`；已驗證舊寫法 RED、修法 GREEN。
  → **關鍵**：觸發條件是「同批 Cabin 有值與空值並存」，pandas 才會把 None 轉回 NaN；單筆全空無法重現。
- [ ] 檢查其他直接 `jsonify(DataFrame.to_dict())` 的端點是否有相同風險。

## 影響評估

- **嚴重度**: HIGH（功能於瀏覽器完全不可用，但無資料損毀、無安全風險）
- **影響範圍**: `src/web/ml_api.py` 批次預測端點；前端 `templates/predict.html` 批次上傳流程。
- **破壞性變更**: 無（回應結構不變，僅 `NaN` → `null`；內部 helper 更名不對外）。
