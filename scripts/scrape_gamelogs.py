"""
scrape_gamelogs.py
Fetches the latest NHL gamelog data from MoneyPuck and appends new games to Supabase.
"""

import os
import io
import pandas as pd
import requests
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.environ['DB_HOST'].strip()
DB_PORT = os.environ['DB_PORT'].strip()
DB_NAME = os.environ['DB_NAME'].strip()
DB_USER = os.environ['DB_USER'].strip()
DB_PASS = os.environ['DB_PASS'].strip()

TABLE_NAME = 'NHL_Gamelog'
URL = 'https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv'

TEAM_ABBREVIATION_MAP = {
    'L.A': 'LAK',
    'N.J': 'NJD',
    'S.J': 'SJS',
    'T.B': 'TBL',
}

def main():
    print("Fetching gamelog CSV from MoneyPuck...")
    response = requests.get(URL)
    response.raise_for_status()
    data = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
    print(f"  Downloaded {len(data)} rows")

    data['gameId'] = data['gameId'].astype(str).str.strip().str.lower()

    for col in ['team', 'playerTeam', 'opposingTeam']:
        if col in data.columns:
            data[col] = data[col].replace(TEAM_ABBREVIATION_MAP)

    conn_str = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require"
    engine = create_engine(conn_str)
    print("Connected to Supabase")

    existing = pd.read_sql(f'SELECT DISTINCT "gameId" FROM "{TABLE_NAME}"', engine)
    existing['gameId'] = existing['gameId'].astype(str).str.strip().str.lower()

    new_data = data[~data['gameId'].isin(existing['gameId'])]
    print(f"  New games to insert: {len(new_data['gameId'].unique())}")

    if not new_data.empty:
        new_data.to_sql(TABLE_NAME, engine, if_exists='append', index=False)
        print(f"  Appended {len(new_data)} rows to {TABLE_NAME}")
    else:
        print("  No new games to insert")

if __name__ == '__main__':
    main()
