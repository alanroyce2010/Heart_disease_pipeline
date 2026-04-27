# Test Plan — Heart Disease MLOps Monitoring Dashboard
# src/streamlit_app.py

## 1. Test Objectives
Verify correctness, robustness, and reliability of:
- Prometheus metric registration and HTTP server startup
- Single-patient prediction via MLflow serve endpoint
- Data drift detection (KS test for numerics, Chi² for categoricals)
- Model info rendering from metrics.json
- Edge cases: missing files, bad API responses, empty uploads

## 2. Scope
In scope : Unit tests (drift logic, Prometheus helpers), integration tests
           (endpoint calls, file I/O), component tests (tab rendering).
Out of scope : End-to-end browser automation, Grafana integration.

## 3. Acceptance Criteria
- All 40 test cases pass with zero failures.
- Coverage ≥ 85% on src/streamlit_app.py (excluding Streamlit UI calls).
- No test takes longer than 5 seconds individually.

## 4. Test Environment
Python  : 3.10+
Pytest  : 7.x
Extras  : pytest-mock, responses, pytest-cov

## 5. Test Case Summary

| ID    | Area           | Description                                          | Expected      |
|-------|----------------|------------------------------------------------------|---------------|
| TC-01 | Prometheus     | Counter registered once (no duplicate error)         | PASS          |
| TC-02 | Prometheus     | Gauge registered once                                | PASS          |
| TC-03 | Prometheus     | Histogram registered once                            | PASS          |
| TC-04 | Predict        | Valid payload → probability returned                 | PASS          |
| TC-05 | Predict        | Endpoint timeout → error message shown               | PASS          |
| TC-06 | Predict        | Endpoint 500 → error message shown                   | PASS          |
| TC-07 | Predict        | prob ≥ 0.6 → risk = High                            | PASS          |
| TC-08 | Predict        | 0.3 ≤ prob < 0.6 → risk = Moderate                 | PASS          |
| TC-09 | Predict        | prob < 0.3 → risk = Low                             | PASS          |
| TC-10 | Predict        | prob ≥ 0.5 → label = Presence                       | PASS          |
| TC-11 | Predict        | prob < 0.5 → label = Absence                        | PASS          |
| TC-12 | Drift          | Identical distributions → no drift                  | PASS          |
| TC-13 | Drift          | Shifted numeric col → KS drift detected             | PASS          |
| TC-14 | Drift          | Shifted categorical col → Chi² drift detected       | PASS          |
| TC-15 | Drift          | All columns drift → fraction = 1.0                  | PASS          |
| TC-16 | Drift          | Batch with missing column → skipped gracefully      | PASS          |
| TC-17 | Drift          | Batch with NaN values → NaN dropped before test     | PASS          |
| TC-18 | Drift          | Single-category batch vs 4-cat ref → Chi² flags drift | PASS        |
| TC-19 | Drift          | Empty batch → exception caught, returns empty lists | PASS          |
| TC-20 | Drift          | Prometheus DRIFT_GAUGE updated after run            | PASS          |
| TC-21 | Drift          | Prometheus DRIFT_SCORE updated after run            | PASS          |
| TC-22 | Drift          | BATCH_SIZE_G set to len(batch)                      | PASS          |
| TC-23 | Metrics file   | metrics.json present → AUC values displayed         | PASS          |
| TC-24 | Metrics file   | metrics.json missing → info message shown           | PASS          |
| TC-25 | Metrics file   | metrics.json malformed JSON → no crash              | PASS          |
| TC-26 | Metrics file   | fold_scores missing → chart not rendered            | PASS          |
| TC-27 | Feature meta   | feature_meta.json present → FEATURES loaded         | PASS          |
| TC-28 | Feature meta   | feature_meta.json missing → fallback columns used   | PASS          |
| TC-29 | Payload build  | Row dict → correct dataframe_split payload shape    | PASS          |
| TC-30 | Payload build  | Column order preserved in payload data              | PASS          |
| TC-31 | Prometheus     | _start_prometheus called twice → OSError swallowed  | PASS          |
| TC-32 | Drift          | ref_df None → batch used as own reference           | PASS          |
| TC-33 | Drift          | KS p-value < 0.05 → drift flag True                 | PASS          |
| TC-34 | Drift          | KS p-value ≥ 0.05 → drift flag False               | PASS          |
| TC-35 | Drift          | Chi² expected zeros filtered by mask                | PASS          |
| TC-36 | Predict        | PRED_COUNTER incremented on success                 | PASS          |
| TC-37 | Predict        | PRED_PROB_HIST observed on success                  | PASS          |
| TC-38 | Drift          | drifted list populated correctly                    | PASS          |
| TC-39 | Drift          | feature_rows length = tested numeric + cat cols     | PASS          |
| TC-40 | Predict        | MLflow JSON response parsed correctly               | PASS          |

## 6. Tools
pytest, pytest-mock, responses (HTTP mocking), pytest-cov
Run: pytest tests/ -v --cov=src --cov-report=term-missing

## 6. Smoke tests (Playwright — requires live app at :8501)

| ID    | Class          | Description                                             | Mark   |
|-------|----------------|---------------------------------------------------------|--------|
| SM-01 | Page load      | Browser tab title contains "Heart Disease"              | smoke  |
| SM-02 | Page load      | Main H1 heading rendered                                | smoke  |
| SM-03 | Page load      | All three tabs present                                  | smoke  |
| SM-04 | Page load      | No Streamlit exception banner on load                   | smoke  |
| SM-05 | Page load      | Prometheus :8502/metrics returns 200                    | smoke  |
| SM-06 | Predict tab    | Heading "Single Patient Prediction" visible             | smoke  |
| SM-07 | Predict tab    | Age, BP, Cholesterol inputs rendered                    | smoke  |
| SM-08 | Predict tab    | Predict button visible and enabled                      | smoke  |
| SM-09 | Predict tab    | Click Predict → metric or error alert, no traceback     | ui     |
| SM-10 | Predict tab    | MLflow endpoint URL caption visible                     | ui     |
| SM-11 | Predict tab    | Age slider is interactive                               | ui     |
| SM-12 | Predict tab    | ST Depression slider rendered                           | ui     |
| SM-13 | Drift tab      | "Data Drift Monitor" heading visible                    | smoke  |
| SM-14 | Drift tab      | CSV file uploader widget present                        | smoke  |
| SM-15 | Drift tab      | KS / Chi2 instructions displayed                        | smoke  |
| SM-16 | Drift tab      | Upload valid CSV → batch size shown                     | ui     |
| SM-17 | Drift tab      | Upload CSV → results table has Feature column           | ui     |
| SM-18 | Drift tab      | No upload → info box, no exception                      | smoke  |
| SM-19 | Drift tab      | Upload CSV → Drifted Features + Drift Fraction visible  | ui     |
| SM-20 | Model tab      | "Model & Pipeline Info" heading visible                 | smoke  |
| SM-21 | Model tab      | Prometheus metrics reference table rendered             | smoke  |
| SM-22 | Model tab      | No exception on Model tab                               | smoke  |
| SM-23 | Model tab      | Metrics OR info prompt shown                            | smoke  |
| SM-24 | Model tab      | Port 8502 mentioned                                     | smoke  |
| SM-25 | Navigation     | Rapid tab switching causes no crash                     | smoke  |
| SM-26 | Navigation     | Hard page reload recovers cleanly                       | smoke  |
| SM-27 | Navigation     | Upload + tab switch + return causes no crash            | ui     |

## 7. Tools
pytest, pytest-mock, responses, pytest-cov
pytest-playwright, playwright (chromium) — smoke tests only

Run unit tests:
    pytest tests/test_streamlit_app.py -v --cov=src --cov-report=term-missing

Run smoke tests (requires live app stack):
    pytest tests/test_smoke.py -v -m smoke
    pytest tests/test_smoke.py -v -m ui
    pytest tests/test_smoke.py -v