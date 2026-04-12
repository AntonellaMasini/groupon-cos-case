"""
Export analysis results to Google Sheets with conditional formatting.
Requires credentials.json (service account) in the project root.
"""

import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CREDS_PATH = ROOT / "credentials.json"
CLEAN_PATH = ROOT / "data" / "tickets_clean.csv"
EXCEL_PATH = ROOT / "output" / "analysis_results.xlsx"
SHEET_NAME = "Groupon Ops Analysis"
SHEET_ID = "1-pGRGnhFRUDiknhWk8vvJC00TcvH7SaPtve7wpXm1lI"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# Conditional formatting colors
RED = {"red": 0.95, "green": 0.8, "blue": 0.8}
GREEN = {"red": 0.8, "green": 0.95, "blue": 0.8}


def get_client():
    creds = Credentials.from_service_account_file(str(CREDS_PATH), scopes=SCOPES)
    return gspread.authorize(creds)


def col_letter(idx):
    """Convert 0-based column index to spreadsheet letter (0=A, 1=B, ...)."""
    result = ""
    idx += 1
    while idx > 0:
        idx, remainder = divmod(idx - 1, 26)
        result = chr(65 + remainder) + result
    return result


def write_tab(spreadsheet, tab_name, df):
    """Write a dataframe to a sheet tab with bold headers and frozen row 1."""
    try:
        worksheet = spreadsheet.worksheet(tab_name)
        worksheet.clear()
        # Reset all background colors to white (clears leftover formatting)
        spreadsheet.batch_update({"requests": [{
            "repeatCell": {
                "range": {"sheetId": worksheet.id},
                "cell": {"userEnteredFormat": {"backgroundColor": {"red": 1, "green": 1, "blue": 1}}},
                "fields": "userEnteredFormat.backgroundColor",
            }
        }]})
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=tab_name, rows=len(df) + 1, cols=len(df.columns) + 1)

    # If index has a name (e.g. assigned_team), include it as a column
    if df.index.name:
        df = df.reset_index()

    # Write headers + data
    headers = df.columns.tolist()
    values = [headers] + df.fillna("").values.tolist()

    # Convert numpy types to native Python and round floats
    for i, row in enumerate(values):
        for j, val in enumerate(row):
            if hasattr(val, "item"):  # numpy scalar → native Python
                val = val.item()
                values[i][j] = val
            if isinstance(val, float):
                values[i][j] = round(val, 2)

    worksheet.update(values, value_input_option="USER_ENTERED")

    # Bold headers
    worksheet.format("1:1", {"textFormat": {"bold": True}})

    # Freeze row 1
    worksheet.freeze(rows=1)

    print(f"  Written: {tab_name} ({len(df)} rows)")
    return worksheet, df


def apply_formatting(worksheet, df):
    """Directly color cells red/green based on metric thresholds via batch update."""
    columns = df.columns.tolist()
    requests = []

    for col_idx, col_name in enumerate(columns):
        col_lower = col_name.lower()
        is_resolution = "resolution_rate" in col_lower
        is_csat = "csat" in col_lower

        if not is_resolution and not is_csat:
            continue

        for row_idx in range(len(df)):
            val = df.iloc[row_idx, col_idx]
            if pd.isna(val):
                continue

            color = None
            if is_resolution and val < 60:
                color = RED
            elif is_resolution and val > 70:
                color = GREEN
            elif is_csat and val < 3.0:
                color = RED
            elif is_csat and val > 3.8:
                color = GREEN

            if color:
                requests.append({
                    "repeatCell": {
                        "range": {
                            "sheetId": worksheet.id,
                            "startRowIndex": row_idx + 1,
                            "endRowIndex": row_idx + 2,
                            "startColumnIndex": col_idx,
                            "endColumnIndex": col_idx + 1,
                        },
                        "cell": {
                            "userEnteredFormat": {"backgroundColor": color}
                        },
                        "fields": "userEnteredFormat.backgroundColor",
                    }
                })

    if requests:
        worksheet.spreadsheet.batch_update({"requests": requests})
        print(f"  Formatted: {len(requests)} cells colored")


def main():
    print("── Google Sheets Export ─────────────────────────────────")

    # Load all tabs from Excel
    tabs_to_export = [
        "Team Performance",
        "Channel Performance",
        "Category Analysis",
        "Weekly Trends",
        "Anomalies",
    ]

    dfs = {}
    for tab in tabs_to_export:
        dfs[tab] = pd.read_excel(EXCEL_PATH, sheet_name=tab, index_col=0
                                 if tab != "Anomalies" else None)

    # Load cleaned ticket data
    clean_df = pd.read_csv(CLEAN_PATH)
    # Drop customer_message to keep the sheet manageable
    clean_df = clean_df.drop(columns=["customer_message"], errors="ignore")
    dfs["Tickets Clean"] = clean_df

    # Connect to Google Sheets
    client = get_client()

    # Open spreadsheet by ID (shared with service account)
    spreadsheet = client.open_by_key(SHEET_ID)
    print(f"  Opened sheet: {spreadsheet.title}")

    # Write each tab
    for tab_name, df in dfs.items():
        worksheet, clean_df = write_tab(spreadsheet, tab_name, df)
        apply_formatting(worksheet, clean_df)

    # Remove default Sheet1 if it exists
    try:
        default = spreadsheet.worksheet("Sheet1")
        spreadsheet.del_worksheet(default)
    except gspread.WorksheetNotFound:
        pass

    print(f"\n  Spreadsheet URL: {spreadsheet.url}")
    print("── Done ────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
