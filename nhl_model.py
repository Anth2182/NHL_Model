"""
nhl_model.py
Shared model classes and helpers used by both the notebook and the Streamlit app.
The notebook handles DB queries and training; this module is for prediction only.
"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize
from scipy.special import factorial

# ── Constants ─────────────────────────────────────────────────────────────────

_HERE        = os.path.dirname(os.path.abspath(__file__))
SCHEDULE_CSV = os.path.join(_HERE, 'nhl-202526-asplayed.csv')
MODEL_PATH   = os.path.join(_HERE, 'nhl_skellam_model.json')
B2B_PENALTY  = -0.04

TEAM_NAME_MAP = {
    'Anaheim Ducks':         'ANA',
    'Boston Bruins':         'BOS',
    'Buffalo Sabres':        'BUF',
    'Calgary Flames':        'CGY',
    'Carolina Hurricanes':   'CAR',
    'Chicago Blackhawks':    'CHI',
    'Colorado Avalanche':    'COL',
    'Columbus Blue Jackets': 'CBJ',
    'Dallas Stars':          'DAL',
    'Detroit Red Wings':     'DET',
    'Edmonton Oilers':       'EDM',
    'Florida Panthers':      'FLA',
    'Los Angeles Kings':     'LAK',
    'Minnesota Wild':        'MIN',
    'Montreal Canadiens':    'MTL',
    'Nashville Predators':   'NSH',
    'New Jersey Devils':     'NJD',
    'New York Islanders':    'NYI',
    'New York Rangers':      'NYR',
    'Ottawa Senators':       'OTT',
    'Philadelphia Flyers':   'PHI',
    'Pittsburgh Penguins':   'PIT',
    'San Jose Sharks':       'SJS',
    'Seattle Kraken':        'SEA',
    'St. Louis Blues':       'STL',
    'Tampa Bay Lightning':   'TBL',
    'Toronto Maple Leafs':   'TOR',
    'Utah Hockey Club':      'UTA',
    'Utah Mammoth':          'UTA',
    'Vancouver Canucks':     'VAN',
    'Vegas Golden Knights':  'VGK',
    'Washington Capitals':   'WSH',
    'Winnipeg Jets':         'WPG',
}

# ── Helper functions ───────────────────────────────────────────────────────────

def dc_rho(goals_h, goals_a, lam_h, lam_a, rho):
    if goals_h == 0 and goals_a == 0:
        return 1 - lam_h * lam_a * rho
    elif goals_h == 1 and goals_a == 0:
        return 1 + lam_a * rho
    elif goals_h == 0 and goals_a == 1:
        return 1 + lam_h * rho
    elif goals_h == 1 and goals_a == 1:
        return 1 - rho
    return 1.0


def poisson_pmf(k, lam):
    return np.exp(-lam) * (lam ** k) / factorial(k)


def _american_to_prob(odds):
    try:
        odds = float(odds)
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
    except (ValueError, TypeError):
        return None


def load_schedule(csv_path=SCHEDULE_CSV):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={'Score': 'visitor_score', 'Score.1': 'home_score'})
    df['Date']          = pd.to_datetime(df['Date']).dt.date
    df['home_abbr']     = df['Home'].map(TEAM_NAME_MAP)
    df['visitor_abbr']  = df['Visitor'].map(TEAM_NAME_MAP)
    return df


# ── Core model classes ─────────────────────────────────────────────────────────

class DixonColesModel:
    def __init__(self):
        self.attack = {}; self.defense = {}
        self.home_adv = None; self.log_mu = None
        self.rho = None; self.teams = []

    def _deduplicate(self, games_df):
        home = games_df[games_df['home_or_away'] == 'HOME'].copy()
        if len(home) == 0:
            seen, rows = set(), []
            for _, r in games_df.iterrows():
                if r['gameId'] not in seen:
                    seen.add(r['gameId']); rows.append(r)
            home = pd.DataFrame(rows)
        return home

    def fit(self, games_df, xi=0.005, reference_date=None):
        self.teams = sorted(games_df['team'].unique().tolist())
        n    = len(self.teams)
        t2i  = {t: i for i, t in enumerate(self.teams)}
        home_games = self._deduplicate(games_df).reset_index(drop=True)

        if xi > 0:
            dates = pd.to_datetime(home_games['gameDate'])
            ref   = pd.to_datetime(reference_date) if reference_date else dates.max()
            days_ago = (ref - dates).dt.days.clip(lower=0).values
            weights  = np.exp(-xi * days_ago)
        else:
            weights = np.ones(len(home_games))

        n_params = 3 + (n - 1) + n

        def params_to_dict(p):
            rho    = np.clip(p[2], -0.99, 0.99)
            attack = np.concatenate([[0.0], p[3:3 + n - 1]])
            return p[0], p[1], rho, attack, p[3 + n - 1:]

        def neg_log_likelihood(p):
            log_mu, home_adv, rho, attack, defense = params_to_dict(p)
            ll = 0.0
            for idx, row in home_games.iterrows():
                hi = t2i.get(row['team']); ai = t2i.get(row['opposingTeam'])
                if hi is None or ai is None: continue
                gh    = int(row['goalsFor']); ga = int(row['goalsAgainst'])
                lam_h = np.clip(np.exp(log_mu + home_adv + attack[hi] - defense[ai]), 0.01, 20)
                lam_a = np.clip(np.exp(log_mu             + attack[ai] - defense[hi]), 0.01, 20)
                dc    = max(dc_rho(gh, ga, lam_h, lam_a, rho), 1e-10)
                game_ll = (np.log(poisson_pmf(gh, lam_h) + 1e-10)
                         + np.log(poisson_pmf(ga, lam_a) + 1e-10)
                         + np.log(dc))
                ll += weights[idx] * game_ll
            return -ll

        avg_goals = home_games['goalsFor'].mean()
        x0 = np.zeros(n_params); x0[0] = np.log(max(avg_goals, 0.5)); x0[1] = 0.1; x0[2] = -0.1
        result = minimize(neg_log_likelihood, x0, method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-6, 'maxfun': 3000})
        log_mu, home_adv, rho, attack_arr, defense_arr = params_to_dict(result.x)
        self.log_mu = log_mu; self.home_adv = home_adv
        self.rho    = np.clip(rho, -0.99, 0.99)
        self.attack  = {t: attack_arr[i]  for i, t in enumerate(self.teams)}
        self.defense = {t: defense_arr[i] for i, t in enumerate(self.teams)}
        return self


def skellam_probabilities(lam_h, lam_a, rho=-0.1, max_goals=15, ou_line=None):
    score_matrix = np.zeros((max_goals + 1, max_goals + 1))
    for gh in range(max_goals + 1):
        for ga in range(max_goals + 1):
            dc   = dc_rho(gh, ga, lam_h, lam_a, rho)
            prob = poisson_pmf(gh, lam_h) * poisson_pmf(ga, lam_a) * dc
            score_matrix[gh, ga] = max(prob, 0)
    total = score_matrix.sum()
    if total > 0:
        score_matrix /= total

    home_win = np.tril(score_matrix, -1).sum()
    draw     = np.trace(score_matrix)
    away_win = np.triu(score_matrix,  1).sum()
    home_total = home_win + draw * 0.50
    away_total = away_win + draw * 0.50

    home_pl_prob = sum(score_matrix[gh, ga]
                       for gh in range(max_goals + 1)
                       for ga in range(max_goals + 1) if gh - ga >= 2)
    away_pl_prob = 1.0 - home_pl_prob

    result = {
        'home_win_prob':  round(home_total,    4),
        'away_win_prob':  round(away_total,    4),
        'home_reg_prob':  round(home_win,      4),
        'away_reg_prob':  round(away_win,      4),
        'ot_prob':        round(draw,          4),
        'home_pl_prob':   round(home_pl_prob,  4),
        'away_pl_prob':   round(away_pl_prob,  4),
        'lam_home':       round(lam_h,         3),
        'lam_away':       round(lam_a,         3),
        'expected_total': round(lam_h + lam_a, 3),
    }

    if ou_line is not None:
        over_prob = under_prob = push_prob = 0.0
        for gh in range(max_goals + 1):
            for ga in range(max_goals + 1):
                t = gh + ga
                if   t > ou_line: over_prob  += score_matrix[gh, ga]
                elif t < ou_line: under_prob += score_matrix[gh, ga]
                else:             push_prob  += score_matrix[gh, ga]
        denom = over_prob + under_prob
        result['over_prob']  = round(over_prob,  4)
        result['under_prob'] = round(under_prob, 4)
        result['push_prob']  = round(push_prob,  4)
        result['over_fair']  = round(over_prob  / denom, 4) if denom > 0 else None
        result['under_fair'] = round(under_prob / denom, 4) if denom > 0 else None

    return result


class CurrentSeasonMetrics:
    def __init__(self, df):
        self.atk_adj = {}; self.def_adj = {}
        if len(df) > 0: self._calculate(df)

    def _calculate(self, df):
        records = []
        for team in df['team'].unique():
            tg = df[df['team'] == team]; n = len(tg)
            records.append({'team': team,
                'xgf_pg':   tg['xGoalsFor'].sum() / n,
                'xga_pg':   tg['xGoalsAgainst'].sum() / n,
                'hdxgf_pg': tg['highDangerxGoalsFor'].sum() / n,
                'hdxga_pg': tg['highDangerxGoalsAgainst'].sum() / n,
                'sogf_pg':  tg['shotsOnGoalFor'].sum() / n,
                'soga_pg':  tg['shotsOnGoalAgainst'].sum() / n,
            })
        mdf = pd.DataFrame(records)

        def zscore(col):
            s = mdf[col].std()
            return (mdf[col] - mdf[col].mean()) / s if s > 0 else 0

        atk_z = 0.50*zscore('xgf_pg') + 0.30*zscore('hdxgf_pg') + 0.20*zscore('sogf_pg')
        def_z = 0.50*zscore('xga_pg') + 0.30*zscore('hdxga_pg') + 0.20*zscore('soga_pg')
        scale = 0.10
        for i, row in mdf.iterrows():
            self.atk_adj[row['team']] = float(atk_z.iloc[i] * scale)
            self.def_adj[row['team']] = float(def_z.iloc[i] * scale)

    def get_atk(self, team): return self.atk_adj.get(team, 0.0)
    def get_def(self, team): return self.def_adj.get(team, 0.0)


class GoalieRatings:
    def __init__(self, goalie_df=None):
        self.ratings = {}
        if goalie_df is not None: self._load(goalie_df)

    def _load(self, df):
        gdf = df[(df['situation'] == 'all') & (df['games_played'] >= 5)].copy()
        gdf['gsax']        = gdf['xGoals'] - gdf['goals']
        gdf['gsax_per_gp'] = gdf['gsax'] / gdf['games_played']
        for _, row in gdf.iterrows():
            gsax_pg = np.clip(row['gsax_per_gp'], -0.40, 0.40)
            self.ratings[row['name']] = {
                'log_adj': gsax_pg * 0.33,
                'gsax':    row['gsax'],
                'gsax_pg': row['gsax_per_gp'],
                'games':   row['games_played'],
                'team':    row['team'],
            }

    def get_log_adj(self, name):
        return self.ratings[name]['log_adj'] if name in self.ratings else 0.0


class NHLPoissonModel:
    def __init__(self, prior_season_df, current_season_df, goalie_df=None, xi=0.005):
        self.xi = xi
        prior_ref = pd.to_datetime(prior_season_df['gameDate']).max()
        self.prior_dc = DixonColesModel().fit(prior_season_df, xi=xi, reference_date=prior_ref)
        self.current_games = len(current_season_df['gameId'].unique())
        if self.current_games >= 10:
            curr_ref = pd.to_datetime(current_season_df['gameDate']).max()
            self.current_dc  = DixonColesModel().fit(current_season_df, xi=xi, reference_date=curr_ref)
            self.has_curr_dc = True
        else:
            self.has_curr_dc = False
        self.metrics = CurrentSeasonMetrics(current_season_df)
        self.goalies = GoalieRatings(goalie_df) if goalie_df is not None else None
        self._calculate_weights()
        self._blend_ratings()
        self.rho = self.current_dc.rho if self.has_curr_dc else self.prior_dc.rho

    def _calculate_weights(self):
        g = self.current_games
        if   g < 10:  self.prior_w = 0.80
        elif g < 20:  self.prior_w = 0.50
        elif g < 30:  self.prior_w = 0.20
        else:         self.prior_w = 0.00
        cw = 1 - self.prior_w
        if   g < 10:  self.curr_wl_w = cw*0.20; self.curr_met_w = cw*0.80
        elif g < 20:  self.curr_wl_w = cw*0.50; self.curr_met_w = cw*0.50
        elif g < 40:  self.curr_wl_w = cw*0.75; self.curr_met_w = cw*0.25
        else:         self.curr_wl_w = cw*0.40; self.curr_met_w = cw*0.60

    def _blend_ratings(self):
        all_teams = set(self.prior_dc.teams)
        if self.has_curr_dc: all_teams.update(self.current_dc.teams)
        all_teams.update(self.metrics.atk_adj.keys())
        self.attack = {}; self.defense = {}
        for team in all_teams:
            p_atk = self.prior_dc.attack.get(team, 0.0)
            p_def = self.prior_dc.defense.get(team, 0.0)
            c_atk = self.current_dc.attack.get(team, 0.0)  if self.has_curr_dc else 0.0
            c_def = self.current_dc.defense.get(team, 0.0) if self.has_curr_dc else 0.0
            self.attack[team]  = self.prior_w*p_atk + self.curr_wl_w*c_atk + self.curr_met_w*self.metrics.get_atk(team)
            self.defense[team] = self.prior_w*p_def + self.curr_wl_w*c_def + self.curr_met_w*self.metrics.get_def(team)
        curr_adv = self.current_dc.home_adv if self.has_curr_dc else self.prior_dc.home_adv
        curr_mu  = self.current_dc.log_mu   if self.has_curr_dc else self.prior_dc.log_mu
        self.home_adv = self.prior_w*self.prior_dc.home_adv + (1-self.prior_w)*curr_adv
        self.log_mu   = self.prior_w*self.prior_dc.log_mu   + (1-self.prior_w)*curr_mu

    def predict(self, home_team, away_team,
                home_goalie=None, away_goalie=None,
                home_b2b=False, away_b2b=False,
                ou_line=None):
        home_goalie_adj = self.goalies.get_log_adj(home_goalie) if (self.goalies and home_goalie) else 0.0
        away_goalie_adj = self.goalies.get_log_adj(away_goalie) if (self.goalies and away_goalie) else 0.0
        lam_h = np.exp(self.log_mu + self.home_adv
                       + self.attack.get(home_team, 0.0)
                       - self.defense.get(away_team, 0.0)
                       - away_goalie_adj)
        lam_a = np.exp(self.log_mu
                       + self.attack.get(away_team, 0.0)
                       - self.defense.get(home_team, 0.0)
                       - home_goalie_adj)
        if home_b2b: lam_h *= np.exp(B2B_PENALTY)
        if away_b2b: lam_a *= np.exp(B2B_PENALTY)
        lam_h = np.clip(lam_h, 0.01, 20)
        lam_a = np.clip(lam_a, 0.01, 20)
        result = skellam_probabilities(lam_h, lam_a, rho=self.rho, ou_line=ou_line)
        result.update({
            'home_team': home_team, 'away_team': away_team,
            'home_goalie': home_goalie or 'N/A', 'away_goalie': away_goalie or 'N/A',
            'home_goalie_adj': round(home_goalie_adj, 4),
            'away_goalie_adj': round(away_goalie_adj, 4),
            'home_b2b': home_b2b, 'away_b2b': away_b2b,
        })
        return result

    def save(self, filename=MODEL_PATH):
        data = {
            'attack': self.attack, 'defense': self.defense,
            'home_adv': self.home_adv, 'log_mu': self.log_mu,
            'rho': self.rho, 'xi': self.xi,
            'current_games': self.current_games,
            'prior_weight': self.prior_w,
            'curr_wl_weight': self.curr_wl_w,
            'curr_metric_weight': self.curr_met_w,
            'goalie_ratings': self.goalies.ratings if self.goalies else {},
            'timestamp': datetime.now().isoformat(),
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filename=MODEL_PATH):
        with open(filename) as f:
            data = json.load(f)
        m = cls.__new__(cls)
        m.attack        = data['attack']
        m.defense       = data['defense']
        m.home_adv      = data['home_adv']
        m.log_mu        = data['log_mu']
        m.rho           = data['rho']
        m.xi            = data.get('xi', 0.005)
        m.current_games = data['current_games']
        m.prior_w       = data['prior_weight']
        m.curr_wl_w     = data['curr_wl_weight']
        m.curr_met_w    = data['curr_metric_weight']
        m.has_curr_dc   = m.current_games >= 10
        m.timestamp     = data.get('timestamp', 'unknown')
        if data.get('goalie_ratings'):
            m.goalies = GoalieRatings()
            m.goalies.ratings = data['goalie_ratings']
        else:
            m.goalies = None
        return m
