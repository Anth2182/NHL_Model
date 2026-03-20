"""
scrape_goalies.py
Fetches current season goalie stats from MoneyPuck and replaces the Supabase table.
"""

import os
import io
import pandas as pd
import requests
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.environ['DB_HOST']
DB_PORT = os.environ['DB_PORT']
DB_NAME = os.environ['DB_NAME']
DB_USER = os.environ['DB_USER']
DB_PASS = os.environ['DB_PASS']

TABLE_NAME = 'Goalie_stats'
SEASON = '2025'
URL = f'https://moneypuck.com/moneypuck/playerData/seasonSummary/{SEASON}/regular/goalies.csv'

def main():
    print(f"Fetching goalie stats for season {SEASON} from MoneyPuck...")
    response = requests.get(URL)
    response.raise_for_status()
    data = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
    print(f"  Downloaded {len(data)} rows")

    if 'playerId' in data.columns:
        data['playerId'] = data['playerId'].astype(str).str.strip().str.lower()

    conn_str = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require"
    engine = create_engine(conn_str)
    print("Connected to Supabase")

    data.to_sql(TABLE_NAME, engine, if_exists='replace', index=False)
    print(f"  Replaced '{TABLE_NAME}' with {len(data)} rows")

if __name__ == '__main__':
    main()
