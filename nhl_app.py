"""
nhl_app.py  —  NHL Prediction Frontend
Run with:  streamlit run nhl_app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from nhl_model import (
    NHLPoissonModel, load_schedule, _american_to_prob,
    MODEL_PATH, SCHEDULE_CSV
)

def _parse_odds(s):
    """Parse a text odds input like '-123' or '+110' to int, or None if blank/invalid."""
    if s is None: return None
    s = str(s).strip()
    if not s: return None
    try: return int(s)
    except ValueError: return None

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NHL Edge Finder",
    page_icon="🏒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 6px 16px; }
</style>
<script>
    // Force full keyboard on mobile so the minus sign (-) is available for odds input
    const _fixInputMode = () => {
        document.querySelectorAll('input[type="text"], input:not([type])').forEach(el => {
            el.setAttribute('inputmode', 'text');
        });
    };
    const _observer = new MutationObserver(_fixInputMode);
    _observer.observe(document.body, { childList: true, subtree: true });
    _fixInputMode();
</script>
""", unsafe_allow_html=True)

# ── Load model + schedule (cached) ────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return NHLPoissonModel.load(MODEL_PATH)

@st.cache_data(show_spinner="Loading schedule...")
def get_schedule():
    return load_schedule(SCHEDULE_CSV)

try:
    model = load_model()
    sched = get_schedule()
except FileNotFoundError as e:
    st.error(f"Could not load required file: {e}\n\nRun the notebook to train and save the model first.")
    st.stop()

# ── Build team → goalie lookup ─────────────────────────────────────────────────

team_goalies: dict = {}
if model.goalies:
    for g, info in model.goalies.ratings.items():
        team = info['team']
        team_goalies.setdefault(team, []).append(g)
    for t in team_goalies:
        team_goalies[t] = sorted(team_goalies[t])

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏒 NHL Edge Finder")
    st.divider()

    st.markdown("**Model Info**")
    st.caption(f"Saved: {getattr(model, 'timestamp', 'unknown')[:10]}")
    st.caption(f"Teams: {len(model.attack)}  |  Goalies: {len(model.goalies.ratings) if model.goalies else 0}")
    st.caption(f"xi: {model.xi}  |  Prior weight: {model.prior_w:.0%}")
    st.caption(f"Current DC: {model.curr_wl_w:.0%}  |  Metrics: {model.curr_met_w:.0%}")

    st.divider()
    st.markdown("**Goalie Ratings**")
    st.caption("Dropdowns are pre-filtered by team. Use the override field for trades/injuries.")
    if model.goalies:
        goalie_names = sorted(model.goalies.ratings.keys())
        st.dataframe(
            pd.DataFrame({'Goalie': goalie_names,
                          'Team':   [model.goalies.ratings[g]['team'] for g in goalie_names],
                          'GSAx/G': [round(model.goalies.ratings[g]['gsax_pg'], 3) for g in goalie_names]}),
            hide_index=True,
            height=400,
        )

    st.divider()
    if st.button("Reload model", use_container_width=True):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

# ── Main ───────────────────────────────────────────────────────────────────────

st.title("NHL Edge Finder")

# Date picker
col_date, col_info = st.columns([2, 3])
with col_date:
    selected_date = st.date_input("Game Date", value=datetime.now(ZoneInfo("America/New_York")).date(), label_visibility="collapsed")

prev_day  = selected_date - timedelta(days=1)
b2b_teams = set(
    sched[sched['Date'] == prev_day]['home_abbr'].dropna().tolist() +
    sched[sched['Date'] == prev_day]['visitor_abbr'].dropna().tolist()
)

day_games = sched[sched['Date'] == selected_date].copy()
day_games = day_games[day_games['home_abbr'].notna() & day_games['visitor_abbr'].notna()]

with col_info:
    if len(day_games) == 0:
        st.warning(f"No games found in schedule for {selected_date}")
        st.stop()
    else:
        b2b_count = sum(1 for _, g in day_games.iterrows()
                        if g['home_abbr'] in b2b_teams or g['visitor_abbr'] in b2b_teams)
        st.info(f"**{selected_date.strftime('%B %d, %Y')}** — {len(day_games)} games"
                + (f"  |  {b2b_count} back-to-back" if b2b_count else ""))

st.divider()

# ── Game input form ────────────────────────────────────────────────────────────

with st.form("predictions_form"):
    game_inputs = {}
    ml_data  = {}
    tot_data = {}

    tab_lines, tab_totals = st.tabs(["📊 Lines", "🎯 Totals"])

    with tab_lines:
        hdr = st.columns([2, 1.2, 1.2, 2.2, 2.2, 0.9])
        for col, label in zip(hdr, ["Matchup", "Home ML", "Away ML", "Home Goalie", "Away Goalie", "B2B"]):
            col.markdown(f"**{label}**")
        st.divider()

        for _, game in day_games.iterrows():
            home = game['home_abbr']
            away = game['visitor_abbr']
            key  = f"{home} vs {away}"
            h_b2b_auto = home in b2b_teams
            a_b2b_auto = away in b2b_teams

            csv_hg = game.get('Home Goalie', '')
            csv_ag = game.get('Visitor Goalie', '')
            csv_hg = csv_hg if pd.notna(csv_hg) and csv_hg != '' else ''
            csv_ag = csv_ag if pd.notna(csv_ag) and csv_ag != '' else ''

            b2b_flag = " 🔄" if (h_b2b_auto or a_b2b_auto) else ""
            cols = st.columns([2, 1.2, 1.2, 2.2, 2.2, 0.9])
            cols[0].markdown(f"**{home}** vs **{away}**{b2b_flag}")
            home_ml = _parse_odds(cols[1].text_input("Home ML", key=f"{key}_hml", placeholder="-150", label_visibility="collapsed"))
            away_ml = _parse_odds(cols[2].text_input("Away ML", key=f"{key}_aml", placeholder="+130", label_visibility="collapsed"))

            h_opts = [''] + team_goalies.get(home, [])
            h_idx  = h_opts.index(csv_hg) if csv_hg in h_opts else 0
            home_goalie = cols[3].selectbox("Home Goalie", h_opts, index=h_idx, key=f"{key}_hg", label_visibility="collapsed") or None

            a_opts = [''] + team_goalies.get(away, [])
            a_idx  = a_opts.index(csv_ag) if csv_ag in a_opts else 0
            away_goalie = cols[4].selectbox("Away Goalie", a_opts, index=a_idx, key=f"{key}_ag", label_visibility="collapsed") or None

            with cols[5]:
                h_b2b = st.checkbox(home[:3], key=f"{key}_hb2b", value=h_b2b_auto)
                a_b2b = st.checkbox(away[:3], key=f"{key}_ab2b", value=a_b2b_auto)

            ml_data[key] = {
                'home': home, 'away': away,
                'home_ml': home_ml, 'away_ml': away_ml,
                'home_goalie': home_goalie, 'away_goalie': away_goalie,
                'home_b2b': h_b2b, 'away_b2b': a_b2b,
            }

    with tab_totals:
        hdr = st.columns([2, 1.5, 1.5, 1.5])
        for col, label in zip(hdr, ["Matchup", "O/U Line", "Over Odds", "Under Odds"]):
            col.markdown(f"**{label}**")
        st.divider()

        for _, game in day_games.iterrows():
            home = game['home_abbr']
            away = game['visitor_abbr']
            key  = f"{home} vs {away}"

            cols = st.columns([2, 1.5, 1.5, 1.5])
            cols[0].markdown(f"**{home}** vs **{away}**")
            ou_line    = cols[1].number_input("O/U", key=f"{key}_ou", value=None, placeholder="5.5",
                                              min_value=0.5, max_value=15.0, step=0.5, label_visibility="collapsed")
            over_odds  = _parse_odds(cols[2].text_input("Over",  key=f"{key}_over",  placeholder="-110", label_visibility="collapsed"))
            under_odds = _parse_odds(cols[3].text_input("Under", key=f"{key}_under", placeholder="-110", label_visibility="collapsed"))

            tot_data[key] = {'ou_line': ou_line, 'over': over_odds, 'under': under_odds}

    for key, data in ml_data.items():
        game_inputs[key] = {**data, **tot_data.get(key, {'ou_line': None, 'over': None, 'under': None})}

    submitted = st.form_submit_button(
        "Run Predictions",
        type="primary",
        use_container_width=True,
    )

# ── Results ────────────────────────────────────────────────────────────────────

def edge_color(val):
    """Style edge cells: green = positive, red = negative, grey = missing."""
    if not isinstance(val, str) or val == '—':
        return 'color: grey'
    try:
        v = float(val.replace('%', '').replace('+', '')) / 100
        if v >= 0.05:    return 'background-color: #1a7a3a; color: white; font-weight: bold'
        elif v >= 0.02:  return 'background-color: #c6efce; color: #276221'
        elif v >= 0:     return 'background-color: #ebf5eb; color: #276221'
        elif v >= -0.03: return 'background-color: #fce4e4; color: #9c1c1c'
        else:            return 'background-color: #f4b8b8; color: #7a0000'
    except:
        return ''

def show_table(df, edge_col='Edge'):
    styled = df.style.applymap(edge_color, subset=[edge_col])
    st.dataframe(styled, hide_index=True, use_container_width=True)

if submitted:
    st.divider()
    st.subheader(f"Predictions — {selected_date.strftime('%B %d, %Y')}")

    ml_rows = []; tot_rows = []

    def vig_free(ml_a, ml_b):
        a = _american_to_prob(ml_a); b = _american_to_prob(ml_b)
        tot = (a or 0) + (b or 0)
        return (a/tot if tot else None), (b/tot if tot else None)

    def best_edge(model_h, model_a, fair_h, fair_a, label_h, label_a):
        eh = round(model_h - fair_h, 4) if fair_h else None
        ea = round(model_a - fair_a, 4) if fair_a else None
        if eh is None and ea is None: return '—', None
        if eh is None: return label_a, ea
        if ea is None: return label_h, eh
        return (label_h, eh) if eh >= ea else (label_a, ea)

    def fmt(v, pct=True):
        if v is None: return '—'
        return f"{v:.1%}" if pct else str(v)

    def fmt_edge(e):
        return f"{e:+.1%}" if e is not None else '—'

    for key, info in game_inputs.items():
        home = info['home']; away = info['away']
        hg   = info['home_goalie']; ag = info['away_goalie']
        ou   = info['ou_line']

        p = model.predict(home, away,
                          home_goalie=hg, away_goalie=ag,
                          home_b2b=info['home_b2b'], away_b2b=info['away_b2b'],
                          ou_line=ou)

        goalie_str = f"{hg or '—'} / {ag or '—'}"
        b2b_str    = ' '.join(filter(None, [
            f"{home} B2B" if info['home_b2b'] else '',
            f"{away} B2B" if info['away_b2b'] else '',
        ])) or '—'

        # Moneyline
        fair_h, fair_a = vig_free(info['home_ml'], info['away_ml'])
        side, edge_val = best_edge(p['home_win_prob'], p['away_win_prob'],
                                   fair_h, fair_a, home, away)
        ml_rows.append({
            'Matchup':    key,
            'Model H%':   fmt(p['home_win_prob']),
            'Model A%':   fmt(p['away_win_prob']),
            'Fair H%':    fmt(fair_h),
            'Fair A%':    fmt(fair_a),
            'Home ML':    info['home_ml'] if info['home_ml'] else '—',
            'Away ML':    info['away_ml'] if info['away_ml'] else '—',
            'xG':         f"{p['lam_home']:.2f}–{p['lam_away']:.2f}",
            'OT%':        fmt(p['ot_prob']),
            'Goalies':    goalie_str,
            'B2B':        b2b_str,
            'Best Side':  side,
            'Edge':       fmt_edge(edge_val),
            '_edge_num':  edge_val if edge_val is not None else -99,
        })

        # Totals
        if ou and p.get('over_fair') is not None:
            fair_over, fair_under = vig_free(info['over'], info['under'])
            tot_side, tot_edge = best_edge(p['over_fair'], p['under_fair'],
                                           fair_over, fair_under, 'Over', 'Under')
        else:
            tot_side, tot_edge = '—', None

        tot_rows.append({
            'Matchup':      key,
            'xG Total':     f"{p['expected_total']:.2f}",
            'O/U Line':     ou if ou else '—',
            'Model Over%':  fmt(p.get('over_fair')),
            'Model Under%': fmt(p.get('under_fair')),
            'Over Odds':    info['over']  if info['over']  else '—',
            'Under Odds':   info['under'] if info['under'] else '—',
            'Goalies':      goalie_str,
            'B2B':          b2b_str,
            'Best Side':    tot_side,
            'Edge':         fmt_edge(tot_edge),
            '_edge_num':    tot_edge if tot_edge is not None else -99,
        })

    def to_df(rows):
        df = pd.DataFrame(rows).sort_values('_edge_num', ascending=False)
        return df.drop(columns=['_edge_num']).reset_index(drop=True)

    tab_ml, tab_tot = st.tabs(["Moneyline", "Totals"])

    with tab_ml:
        show_table(to_df(ml_rows))

    with tab_tot:
        show_table(to_df(tot_rows))
