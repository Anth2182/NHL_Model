"""
save_predictions.py
Fetches today's NHL odds from The Odds API, runs model predictions,
and saves results to the predictions table in Supabase.
"""

import os
import sys
import requests
import pandas as pd
from datetime import datetime, timezone
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nhl_model import NHLPoissonModel, MODEL_PATH, TEAM_NAME_MAP

load_dotenv()

DB_HOST      = os.environ['DB_HOST'].strip()
DB_PORT      = os.environ['DB_PORT'].strip()
DB_NAME      = os.environ['DB_NAME'].strip()
DB_USER      = os.environ['DB_USER'].strip()
DB_PASS      = os.environ['DB_PASS'].strip()
ODDS_API_KEY = os.environ['ODDS_API_KEY'].strip()

ODDS_URL   = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds/"
TABLE_NAME = "predictions"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    game_date       DATE,
    home_team       VARCHAR(10),
    away_team       VARCHAR(10),
    home_win_prob   FLOAT,
    away_win_prob   FLOAT,
    home_pl_prob    FLOAT,
    away_pl_prob    FLOAT,
    over_prob       FLOAT,
    under_prob      FLOAT,
    expected_total  FLOAT,
    market_home_ml  INTEGER,
    market_away_ml  INTEGER,
    market_ou_line  FLOAT,
    actual_home_goals INTEGER,
    actual_away_goals INTEGER,
    created_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE(game_date, home_team, away_team)
)
"""

def fetch_odds():
    params = {
        'apiKey':      ODDS_API_KEY,
        'regions':     'us',
        'markets':     'h2h,spreads,totals',
        'oddsFormat':  'american',
        'dateFormat':  'iso',
    }
    resp = requests.get(ODDS_URL, params=params, timeout=30)
    resp.raise_for_status()
    remaining = resp.headers.get('x-requests-remaining', '?')
    print(f"  Odds API requests remaining: {remaining}")
    return resp.json()

def extract_market(bookmakers, market_key, preferred_books=('draftkings', 'fanduel', 'bovada')):
    for preferred in preferred_books:
        for book in bookmakers:
            if book['key'] == preferred:
                for market in book['markets']:
                    if market['key'] == market_key:
                        return market['outcomes']
    # fallback to first available
    for book in bookmakers:
        for market in book['markets']:
            if market['key'] == market_key:
                return market['outcomes']
    return None

def main():
    model = NHLPoissonModel.load(MODEL_PATH)
    print(f"Model loaded (trained: {getattr(model, 'timestamp', 'unknown')})")

    conn_str = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require"
    engine = create_engine(conn_str)

    with engine.connect() as conn:
        conn.execute(text(CREATE_TABLE_SQL))
        conn.commit()

    print("Fetching odds from The Odds API...")
    games = fetch_odds()
    print(f"  Found {len(games)} upcoming NHL games")

    rows = []
    for game in games:
        home_full = game['home_team']
        away_full = game['away_team']
        home_abbr = TEAM_NAME_MAP.get(home_full)
        away_abbr = TEAM_NAME_MAP.get(away_full)

        if not home_abbr or not away_abbr:
            print(f"  Skipping unknown team names: {home_full} vs {away_full}")
            continue

        game_date = datetime.fromisoformat(
            game['commence_time'].replace('Z', '+00:00')
        ).astimezone(timezone.utc).date()

        bookmakers = game.get('bookmakers', [])
        h2h    = extract_market(bookmakers, 'h2h')
        totals = extract_market(bookmakers, 'totals')

        home_ml = away_ml = ou_line = None
        if h2h:
            for o in h2h:
                if o['name'] == home_full:  home_ml = int(o['price'])
                elif o['name'] == away_full: away_ml = int(o['price'])
        if totals:
            for o in totals:
                if o['name'] == 'Over': ou_line = float(o['point'])

        pred = model.predict(home_abbr, away_abbr, ou_line=ou_line)

        rows.append({
            'game_date':      str(game_date),
            'home_team':      home_abbr,
            'away_team':      away_abbr,
            'home_win_prob':  pred['home_win_prob'],
            'away_win_prob':  pred['away_win_prob'],
            'home_pl_prob':   pred['home_pl_prob'],
            'away_pl_prob':   pred['away_pl_prob'],
            'over_prob':      pred.get('over_prob'),
            'under_prob':     pred.get('under_prob'),
            'expected_total': pred['expected_total'],
            'market_home_ml': home_ml,
            'market_away_ml': away_ml,
            'market_ou_line': ou_line,
        })
        print(f"  {away_abbr} @ {home_abbr} on {game_date} — "
              f"home win: {pred['home_win_prob']:.1%}, total: {pred['expected_total']:.2f}")

    if not rows:
        print("No games to save.")
        return

    df = pd.DataFrame(rows)
    inserted = 0
    with engine.connect() as conn:
        for _, row in df.iterrows():
            result = conn.execute(text(f"""
                INSERT INTO {TABLE_NAME}
                    (game_date, home_team, away_team,
                     home_win_prob, away_win_prob, home_pl_prob, away_pl_prob,
                     over_prob, under_prob, expected_total,
                     market_home_ml, market_away_ml, market_ou_line)
                VALUES
                    (:game_date, :home_team, :away_team,
                     :home_win_prob, :away_win_prob, :home_pl_prob, :away_pl_prob,
                     :over_prob, :under_prob, :expected_total,
                     :market_home_ml, :market_away_ml, :market_ou_line)
                ON CONFLICT (game_date, home_team, away_team) DO NOTHING
            """), row.to_dict())
            inserted += result.rowcount
        conn.commit()

    print(f"\nSaved {inserted} new predictions to Supabase ({len(rows) - inserted} already existed)")

if __name__ == '__main__':
    main()
