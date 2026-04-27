"""
tests/test_smoke.py — Playwright smoke tests for the Streamlit dashboard.

Requires the full app stack running before pytest is invoked:
    mlflow server --host 127.0.0.1 --port 5000 &
    mlflow models serve -m "models:/heart_disease_best/Production" -p 8001 --no-conda &
    PIPELINE_DIR=. streamlit run src/streamlit_app.py --server.port 8501 &

Install:
    pip install pytest-playwright playwright
    playwright install chromium

Run:
    pytest tests/test_smoke.py -v --base-url http://localhost:8501

Marks:
    @pytest.mark.smoke  — basic page-load and tab checks
    @pytest.mark.ui     — interaction and form tests
"""

import re
import time

import pytest
from playwright.sync_api import Page, expect

BASE_URL = "http://localhost:8501"

# ── Helpers ────────────────────────────────────────────────────────────────

def wait_for_streamlit(page: Page, timeout: int = 15_000):
    """Wait until Streamlit finishes its initial render (spinner gone)."""
    page.wait_for_load_state("networkidle", timeout=timeout)
    # Streamlit shows a top-bar progress indicator while loading
    page.wait_for_selector(
        "[data-testid='stAppViewContainer']",
        state="visible",
        timeout=timeout,
    )
    # Allow any in-flight reruns to settle
    time.sleep(1.0)


def go_to_tab(page: Page, label: str):
    """Click a Streamlit tab by its visible label text."""
    page.get_by_role("tab", name=re.compile(label, re.IGNORECASE)).click()
    time.sleep(0.5)


# ════════════════════════════════════════════════════════════════════════════
# SM-01 – SM-05  Page load & chrome
# ════════════════════════════════════════════════════════════════════════════

class TestPageLoad:

    @pytest.mark.smoke
    def test_sm01_page_title(self, page: Page):
        """SM-01: Browser tab title contains the app name."""
        page.goto(BASE_URL)
        expect(page).to_have_title(re.compile(r"Heart Disease", re.IGNORECASE), timeout=15_000)

    @pytest.mark.smoke
    def test_sm02_main_heading_visible(self, page: Page):
        """SM-02: Main H1 heading is rendered."""
        page.goto(BASE_URL)
        wait_for_streamlit(page)
        heading = page.get_by_text(re.compile(r"Heart Disease Pipeline", re.IGNORECASE))
        expect(heading).to_be_visible()

    @pytest.mark.smoke
    def test_sm03_three_tabs_present(self, page: Page):
        """SM-03: All three tabs — Predict, Drift Monitor, Model Info — are present."""
        page.goto(BASE_URL)
        wait_for_streamlit(page)
        for label in ["Predict", "Drift", "Model"]:
            expect(
                page.get_by_role("tab", name=re.compile(label, re.IGNORECASE))
            ).to_be_visible()

    @pytest.mark.smoke
    def test_sm04_no_uncaught_exception_banner(self, page: Page):
        """SM-04: Streamlit error banner ('Error', 'Exception') is absent on load."""
        page.goto(BASE_URL)
        wait_for_streamlit(page)
        # Streamlit surfaces unhandled exceptions in a red alert box
        error_locator = page.locator("[data-testid='stException']")
        expect(error_locator).to_have_count(0)

    @pytest.mark.smoke
    def test_sm05_prometheus_metrics_endpoint(self, page: Page):
        """SM-05: Prometheus /metrics endpoint at :8502 responds with 200."""
        resp = page.request.get("http://localhost:8502/metrics")
        assert resp.status == 200
        body = resp.text()
        assert "streamlit_predictions_total" in body


# ════════════════════════════════════════════════════════════════════════════
# SM-06 – SM-12  Predict tab
# ════════════════════════════════════════════════════════════════════════════

class TestPredictTab:

    @pytest.mark.smoke
    def test_sm06_predict_tab_content(self, page: Page):
        """SM-06: Predict tab shows the patient form heading."""
        page.goto(BASE_URL)
        wait_for_streamlit(page)
        go_to_tab(page, "Predict")
        expect(
            page.get_by_text(re.compile(r"Single Patient Prediction", re.IGNORECASE))
        ).to_be_visible()

    @pytest.mark.smoke
    def test_sm07_predict_form_inputs_present(self, page: Page):
        """SM-07: Key form inputs (Age slider, BP, Cholesterol) are rendered."""
        page.goto(BASE_URL)
        wait_for_streamlit(page)
        go_to_tab(page, "Predict")
        expect(page.get_by_text("Age")).to_be_visible()
        expect(page.get_by_text(re.compile(r"BP", re.IGNORECASE))).to_be_visible()
        expect(page.get_by_text(re.compile(r"Cholesterol", re.IGNORECASE))).to_be_visible()

    @pytest.mark.smoke
    def test_sm08_predict_button_present(self, page: Page):
        """SM-08: The Predict button is visible and enabled."""
        page.goto(BASE_URL)
        wait_for_streamlit(page)
        go_to_tab(page, "Predict")
        btn = page.get_by_role("button", name=re.compile(r"Predict", re.IGNORECASE))
        expect(btn).to_be_visible()
        expect(btn).to_be_enabled()

    @pytest.mark.ui
    def test_sm09_predict_button_click_shows_result_or_error(self, page: Page):
        """SM-09: Clicking Predict either shows a probability metric or an error
        message — never a blank page or Python traceback."""
        page.goto(BASE_URL)
        wait_for_streamlit(page)
        go_to_tab(page, "Predict")
        page.get_by_role("button", name=re.compile(r"Predict", re.IGNORECASE)).click()
        # Allow the spinner to resolve
        time.sleep(4)
        # Either a metric value or an error alert must appear — not nothing
        metric     = page.locator("[data-testid='stMetric']")
        alert      = page.locator("[data-testid='stAlert']")
        has_metric = metric.count() > 0
        has_alert  = alert.count() > 0
        assert has_metric or has_alert, (
            "Expected either a prediction metric or an error alert after clicking Predict"
        )
        # No unhandled Python exception
        expect(page.locator("[data-testid='stException']")).to_have_count(0)

    @pytest.mark.ui
    def test_sm10_mlflow_endpoint_caption_visible(self, page: Page):
        """SM-10: The MLflow endpoint URL caption is displayed under the heading."""
        page.goto(BASE_URL)
        wait_for_streamlit(page)
        go_to_tab(page, "Predict")
        expect(
            page.get_by_text(re.compile(r"invocations", re.IGNORECASE))
        ).to_be_visible()

    @pytest.mark.ui
    def test_sm11_age_slider_adjustable(self, page: Page):
        """SM-11: The Age slider is interactive — value changes on keyboard input."""
        page.goto(BASE_URL)
        wait_for_streamlit(page)
        go_to_tab(page, "Predict")
        slider = page.locator("[data-testid='stSlider']").first
        expect(slider).to_be_visible()
        # Focus and nudge value
        slider.click()
        page.keyboard.press("ArrowRight")
        time.sleep(0.3)
        # No exception after interaction
        expect(page.locator("[data-testid='stException']")).to_have_count(0)

    @pytest.mark.ui
    def test_sm12_st_depression_slider_present(self, page: Page):
        """SM-12: ST Depression slider is rendered in the Predict tab."""
        page.goto(BASE_URL)
        wait_for_streamlit(page)
        go_to_tab(page, "Predict")
        expect(
            page.get_by_text(re.compile(r"ST Depression", re.IGNORECASE))
        ).to_be_visible()


# ════════════════════════════════════════════════════════════════════════════
# SM-13 – SM-19  Drift Monitor tab
# ════════════════════════════════════════════════════════════════════════════

class TestDriftTab:

    @pytest.mark.smoke
    def test_sm13_drift_tab_heading(self, page: Page):
        """SM-13: Drift Monitor tab heading is visible."""
        page.goto(BASE_URL)
        wait_for_streamlit(page)
        go_to_tab(page, "Drift")
        expect(
            page.get_by_text(re.compile(r"Data Drift Monitor", re.IGNORECASE))
        ).to_be_visible()

    @pytest.mark.smoke
    def test_sm14_file_uploader_present(self, page: Page):
        """SM-14: CSV file uploader widget is present in the Drift tab."""
        page.goto(BASE_URL)
        wait_for_streamlit(page)
        go_to_tab(page, "Drift")
        uploader = page.locator("[data-testid='stFileUploader']")
        expect(uploader).to_be_visible()

    @pytest.mark.smoke
    def test_sm15_instructions_visible(self, page: Page):
        """SM-15: The drift instructions (KS / Chi² explanation) are displayed."""
        page.goto(BASE_URL)
        wait_for_streamlit(page)
        go_to_tab(page, "Drift")
        expect(page.get_by_text(re.compile(r"KS", re.IGNORECASE))).to_be_visible()
        expect(page.get_by_text(re.compile(r"Chi", re.IGNORECASE))).to_be_visible()

    @pytest.mark.ui
    def test_sm16_upload_csv_shows_results(self, page: Page, tmp_path):
        """SM-16: Uploading a valid CSV triggers the drift table to appear."""
        import csv, pathlib

        # Write a minimal valid batch CSV
        csv_file = tmp_path / "batch.csv"
        cols = [
            "Age", "BP", "Cholesterol", "Max HR", "ST depression",
            "Sex", "Chest pain type", "FBS over 120", "EKG results",
            "Exercise angina", "Slope of ST", "Number of vessels fluro", "Thallium",
        ]
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for _ in range(30):
                writer.writerow({
                    "Age": 55, "BP": 130, "Cholesterol": 230, "Max HR": 150,
                    "ST depression": 1.5, "Sex": 1, "Chest pain type": 2,
                    "FBS over 120": 0, "EKG results": 0, "Exercise angina": 0,
                    "Slope of ST": 2, "Number of vessels fluro": 0, "Thallium": 3,
                })

        page.goto(BASE_URL)
        wait_for_streamlit(page)
        go_to_tab(page, "Drift")

        page.locator("[data-testid='stFileUploader'] input[type='file']").set_input_files(
            str(csv_file)
        )
        time.sleep(4)  # allow Streamlit rerun

        # Drift results table or metric should appear
        expect(
            page.get_by_text(re.compile(r"Batch size", re.IGNORECASE))
        ).to_be_visible(timeout=10_000)

    @pytest.mark.ui
    def test_sm17_drift_table_has_feature_column(self, page: Page, tmp_path):
        """SM-17: After upload, the results dataframe contains a 'Feature' column.
        The <th> is present in the DOM but may be scrolled off-screen inside
        Streamlit's sticky-header table -- check existence + text content instead
        of to_be_visible(), which fails on off-viewport elements."""
        import csv

        csv_file = tmp_path / "batch2.csv"
        cols = [
            "Age", "BP", "Cholesterol", "Max HR", "ST depression",
            "Sex", "Chest pain type", "FBS over 120", "EKG results",
            "Exercise angina", "Slope of ST", "Number of vessels fluro", "Thallium",
        ]
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for _ in range(30):
                writer.writerow({c: 1 for c in cols})

        page.goto(BASE_URL)
        wait_for_streamlit(page)
        go_to_tab(page, "Drift")
        page.locator("[data-testid='stFileUploader'] input[type='file']").set_input_files(
            str(csv_file)
        )
        time.sleep(4)

        # Wait for the dataframe to appear at all
        page.wait_for_selector("[data-testid='stDataFrame']", timeout=10_000)

        # Scroll the dataframe into view so the header is in the viewport
        df_el = page.locator("[data-testid='stDataFrame']").first
        df_el.scroll_into_view_if_needed()
        time.sleep(0.5)

        # Check column header exists in DOM and has correct text content.
        # Streamlit virtualises off-screen rows/cols so to_be_visible() is unreliable.
        th = page.locator("th[role='columnheader']").filter(
            has_text=re.compile(r"^Feature$", re.IGNORECASE)
        ).first
        expect(th).to_have_count(1)
        assert "Feature" in (th.text_content() or "")

    @pytest.mark.smoke
    def test_sm18_no_exception_without_upload(self, page: Page):
        """SM-18: Drift tab without any upload shows info box, no exception."""
        page.goto(BASE_URL)
        wait_for_streamlit(page)
        go_to_tab(page, "Drift")
        expect(page.locator("[data-testid='stException']")).to_have_count(0)
        # Info/instructions block should be visible
        expect(
            page.get_by_text(re.compile(r"upload", re.IGNORECASE)).first
        ).to_be_visible()

    @pytest.mark.ui
    def test_sm19_drift_metrics_visible_after_upload(self, page: Page, tmp_path):
        """SM-19: After upload, Drifted Features and Drift Fraction metrics appear."""
        import csv

        csv_file = tmp_path / "batch3.csv"
        cols = [
            "Age", "BP", "Cholesterol", "Max HR", "ST depression",
            "Sex", "Chest pain type", "FBS over 120", "EKG results",
            "Exercise angina", "Slope of ST", "Number of vessels fluro", "Thallium",
        ]
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for _ in range(40):
                writer.writerow({
                    "Age": 55, "BP": 130, "Cholesterol": 230, "Max HR": 150,
                    "ST depression": 1.5, "Sex": 1, "Chest pain type": 2,
                    "FBS over 120": 0, "EKG results": 0, "Exercise angina": 0,
                    "Slope of ST": 2, "Number of vessels fluro": 0, "Thallium": 3,
                })

        page.goto(BASE_URL)
        wait_for_streamlit(page)
        go_to_tab(page, "Drift")
        page.locator("[data-testid='stFileUploader'] input[type='file']").set_input_files(
            str(csv_file)
        )
        time.sleep(5)

        # Use stMetric label locators to avoid strict-mode multi-match.
        # Streamlit renders metric labels as <p> inside [data-testid="stMetricLabel"],
        # which is distinct from table cell text that also contains these words.
        drifted_metric = page.locator("[data-testid='stMetricLabel']").filter(
            has_text=re.compile(r"Drifted Features", re.IGNORECASE)
        ).first
        fraction_metric = page.locator("[data-testid='stMetricLabel']").filter(
            has_text=re.compile(r"Drift Fraction", re.IGNORECASE)
        ).first

        expect(drifted_metric).to_be_visible(timeout=10_000)
        expect(fraction_metric).to_be_visible(timeout=10_000)


# ════════════════════════════════════════════════════════════════════════════
# SM-20 – SM-24  Model Info tab
# ════════════════════════════════════════════════════════════════════════════

class TestModelInfoTab:

    @pytest.mark.smoke
    def test_sm20_model_info_tab_heading(self, page: Page):
        """SM-20: Model Info tab heading is visible."""
        page.goto(BASE_URL)
        wait_for_streamlit(page)
        go_to_tab(page, "Model")
        expect(
            page.get_by_text(re.compile(r"Model.*Pipeline Info", re.IGNORECASE))
        ).to_be_visible()

    @pytest.mark.smoke
    def test_sm21_prometheus_table_visible(self, page: Page):
        """SM-21: The Prometheus metrics reference table is rendered."""
        page.goto(BASE_URL)
        wait_for_streamlit(page)
        go_to_tab(page, "Model")
        expect(
            page.get_by_text(re.compile(r"streamlit_predictions_total", re.IGNORECASE))
        ).to_be_visible()

    @pytest.mark.smoke
    def test_sm22_no_exception_model_tab(self, page: Page):
        """SM-22: Model Info tab loads without Python exceptions."""
        page.goto(BASE_URL)
        wait_for_streamlit(page)
        go_to_tab(page, "Model")
        expect(page.locator("[data-testid='stException']")).to_have_count(0)

    @pytest.mark.smoke
    def test_sm23_metrics_or_info_message(self, page: Page):
        """SM-23: Either metrics (if pipeline has run) or an info prompt is shown."""
        page.goto(BASE_URL)
        wait_for_streamlit(page)
        go_to_tab(page, "Model")
        # One of these must be present
        metrics_present = page.locator("[data-testid='stMetric']").count() > 0
        info_present    = page.locator("[data-testid='stAlert']").count() > 0
        assert metrics_present or info_present

    @pytest.mark.smoke
    def test_sm24_prometheus_port_mentioned(self, page: Page):
        """SM-24: Port 8502 is mentioned in the Model Info tab."""
        page.goto(BASE_URL)
        wait_for_streamlit(page)
        go_to_tab(page, "Model")
        expect(page.get_by_text(re.compile(r"8502"))).to_be_visible()


# ════════════════════════════════════════════════════════════════════════════
# SM-25 – SM-27  Navigation & resilience
# ════════════════════════════════════════════════════════════════════════════

class TestNavigation:

    @pytest.mark.smoke
    def test_sm25_tab_switching_no_crash(self, page: Page):
        """SM-25: Rapidly switching between all three tabs causes no crash."""
        page.goto(BASE_URL)
        wait_for_streamlit(page)
        for label in ["Drift", "Model", "Predict", "Drift", "Predict"]:
            go_to_tab(page, label)
        expect(page.locator("[data-testid='stException']")).to_have_count(0)

    @pytest.mark.smoke
    def test_sm26_page_reload_stable(self, page: Page):
        """SM-26: Hard reload of the page recovers cleanly."""
        page.goto(BASE_URL)
        wait_for_streamlit(page)
        page.reload()
        wait_for_streamlit(page)
        expect(
            page.get_by_text(re.compile(r"Heart Disease Pipeline", re.IGNORECASE))
        ).to_be_visible()

    @pytest.mark.ui
    def test_sm27_drift_tab_upload_then_switch_no_crash(self, page: Page, tmp_path):
        """SM-27: Upload a file in Drift tab, switch to Predict, switch back — no crash."""
        import csv

        csv_file = tmp_path / "nav_batch.csv"
        cols = [
            "Age", "BP", "Cholesterol", "Max HR", "ST depression",
            "Sex", "Chest pain type", "FBS over 120", "EKG results",
            "Exercise angina", "Slope of ST", "Number of vessels fluro", "Thallium",
        ]
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for _ in range(20):
                writer.writerow({c: 1 for c in cols})

        page.goto(BASE_URL)
        wait_for_streamlit(page)
        go_to_tab(page, "Drift")
        page.locator("[data-testid='stFileUploader'] input[type='file']").set_input_files(
            str(csv_file)
        )
        time.sleep(2)
        go_to_tab(page, "Predict")
        time.sleep(0.5)
        go_to_tab(page, "Drift")
        time.sleep(0.5)
        expect(page.locator("[data-testid='stException']")).to_have_count(0)