import os
import requests
import zipfile
from io import BytesIO
from datetime import datetime, timedelta
from tqdm import tqdm

BASE_URL = "https://data.binance.vision/data/futures/um/daily/aggTrades"
SYMBOL = "MASKUSDT"  # Change this to your desired trading pair
DOWNLOAD_DIR = "data"
START_DATE = "2025-07-06"
END_DATE = "2025-07-08"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def daterange(start_date, end_date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)

def generate_url(symbol, date):
    date_str = date.strftime("%Y-%m-%d")
    filename = f"{symbol}-{ 'aggTrades' }-{date_str}.zip"
    return f"{BASE_URL}/{symbol}/{filename}", filename

def download_and_extract_zip(url, filename):
    target_csv = filename.replace(".zip", ".csv")
    target_path = os.path.join(DOWNLOAD_DIR, target_csv)

    if os.path.exists(target_path):
        print(f"Already exists: {target_csv}")
        return

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with zipfile.ZipFile(BytesIO(response.content)) as z:
                z.extractall(DOWNLOAD_DIR)
            print(f"Downloaded and extracted: {filename}")
        else:
            print(f"Failed to download (status {response.status_code}): {filename}")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

def main(start, end, symbol):
    start = datetime.strptime(start, "%Y-%m-%d")
    end = datetime.strptime(end, "%Y-%m-%d")

    for date in tqdm(list(daterange(start, end)), desc="Processing dates"):
        url, filename = generate_url(symbol, date)
        download_and_extract_zip(url, filename)

if __name__ == "__main__":
    main()
