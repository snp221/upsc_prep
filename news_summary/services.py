import gspread
from oauth2client.service_account import ServiceAccountCredentials
from collections import defaultdict
from datetime import datetime
import os
import json

def get_google_sheet_data():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    
    # Try environment variable first (for deployment), then file (for local dev)
    google_creds_json = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
    if google_creds_json:
        # Parse JSON from environment variable
        creds_dict = json.loads(google_creds_json)
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    else:
        # Fallback to local file for development
        creds = ServiceAccountCredentials.from_json_keyfile_name('news_summary/gsheet.json', scope)
    
    client = gspread.authorize(creds)

    # Open the Google Sheet by name and select the first sheet
    sheet = client.open('news_summary').sheet1

    # Get all records from the sheet
    data = sheet.get_all_records()
    return data

# Organize data by date
def organize_data_by_date(data):
    date_dict = defaultdict(list)
    for row in data:
        date_str = row['Date']  # Assume 'Date' is the column name for date
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")  # Adjust format if needed
        date_key = date_obj.strftime("%Y-%m-%d")
        date_dict[date_key].append(row)
    return date_dict

