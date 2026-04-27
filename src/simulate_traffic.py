import pandas as pd
import time
from playwright.sync_api import sync_playwright

# ── Configuration ──────────────────────────────────────────────────────────
CSV_FILE = "test.csv"
STREAMLIT_URL = "http://localhost:8501"

def simulate_streamlit_traffic_from_csv(csv_path):
    print(f"Loading data from {csv_path}...\n")
    df = pd.read_csv(csv_path)
    
    with sync_playwright() as p:
        # Launch browser. headless=False lets you visually watch the automation!
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        
        print(f"🌐 Connecting to Streamlit at {STREAMLIT_URL}...")
        page.goto(STREAMLIT_URL)
        
        # Wait for the app to fully load
        page.wait_for_selector('button:has-text("🔬 Predict")', timeout=15000)
        print("✅ App loaded! Starting CSV traffic simulation...\n")
        
        for index, row in df.iterrows():
            patient_id = int(row["id"])
            print(f"🚀 Injecting Patient ID: {patient_id} (Row {index + 1}/{len(df)})...")
            
            try:
                # ── 1. Fill Number Inputs ──
                # Playwright can target Streamlit number inputs by their labels
                page.get_by_label("BP (mmHg)").fill(str(row["BP"]))
                page.get_by_label("Cholesterol (mg/dl)").fill(str(row["Cholesterol"]))
                page.get_by_label("Max HR").fill(str(row["Max HR"]))
                
                sex_text   = "Female" if row["Sex"] == 0 else "Male"
                fbs_text   = "No" if row["FBS over 120"] == 0 else "Yes"
                ang_text   = "No" if row["Exercise angina"] == 0 else "Yes"
                
                # Isolate the exact radio widget using Streamlit's data-testid, then click the option
                page.locator('[data-testid="stRadio"]').filter(has_text="Sex").get_by_text(sex_text, exact=True).click(force=True)
                page.locator('[data-testid="stRadio"]').filter(has_text="FBS > 120").get_by_text(fbs_text, exact=True).click(force=True)
                page.locator('[data-testid="stRadio"]').filter(has_text="Exercise Angina").get_by_text(ang_text, exact=True).click(force=True)

                # Note: Streamlit's custom sliders (Age, ST Depression) and selectboxes 
                # are skipped here to prevent UI coordinate breakage, but the above 
                # inputs are more than enough to create dynamic, varied traffic!

                # ── 3. Click Predict ──
                page.click('button:has-text("🔬 Predict")', force=True)
                
                # ── 4. Wait for processing ──
                # Wait for the Streamlit probability metric to render
                page.wait_for_selector('text=Probability', timeout=10000)
                print("  ✅ Prediction Successful")
                
                # Slight pause to pace the traffic for Grafana
                time.sleep(2)
                
            except Exception as e:
                print(f"  ❌ Error on row {index}: {e}")

        print("\n🎉 CSV Traffic simulation complete!")
        time.sleep(2) # Give a moment to view the final result before closing
        browser.close()

if __name__ == "__main__":
    simulate_streamlit_traffic_from_csv(CSV_FILE)