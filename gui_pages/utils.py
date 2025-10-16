from datetime import datetime, timedelta
import zoneinfo

LOBSTER_DATASETS = {
    "AMZN": {"levels": [1, 5, 10], "date": "2012-06-21"},
    "AAPL": {"levels": [1, 5, 10, 30, 50], "date": "2012-06-21"},
    "GOOG": {"levels": [1, 5, 10], "date": "2012-06-21"},
    "INTC": {"levels": [1, 5, 10], "date": "2012-06-21"},
    "MSFT": {"levels": [1, 5, 10, 30, 50], "date": "2012-06-21"},
    "SPY": {"levels": [30, 50], "date": "2012-06-21"},
}

def seconds_to_eastern_time(seconds: float, date_str: str = "2012-06-21") -> str:
    eastern = zoneinfo.ZoneInfo("America/New_York")
    base_date = datetime.strptime(date_str, "%Y-%m-%d")
    timestamp = base_date + timedelta(seconds=seconds)
    timestamp_eastern = timestamp.replace(tzinfo=eastern)
    return timestamp_eastern.strftime("%H:%M:%S.%f")[:-3]

def get_dataset_date(ticker_name: str) -> str:
    for ticker, info in LOBSTER_DATASETS.items():
        if ticker in ticker_name:
            return info["date"]
    return "2012-06-21"  # Default fallback
