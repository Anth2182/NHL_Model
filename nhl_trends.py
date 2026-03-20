"""
nhl_trends.py  —  NHL Trending Teams Dashboard
Run with:  streamlit run nhl_trends.py

Data source: [dbo].[NHL_Gamelog]  (MoneyPuck game-by-game team data)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text

# ── Config ──────────────────────────────────────────────────────────────────

DB_HOST  = "db.wluxnhuyfumkannjvgiu.supabase.co"
DB_PORT  = "5432"
DB_NAME  = "postgres"
DB_USER  = "postgres"
DB_PASS  = "BlueISland$$87"
SEASON   = 2025

st.set_page_config(
    page_title="NHL Trends",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner="Loading gamelog from Supabase...")
def load_gamelog(season: int = SEASON) -> pd.DataFrame:
    conn_str = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(conn_str, connect_args={"sslmode": "require"})
    q = text("""
        SELECT *
        FROM "NHL_Gamelog"
        WHERE season = :season
          AND situation = 'all'
        ORDER BY gameDate ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={"season": season})

    df['gameDate'] = pd.to_datetime(df['gameDate'])

    # Handle MoneyPuck naming: 'goalsFor' vs 'goals'
    if 'goalsFor' in df.columns and 'goals' not in df.columns:
        df = df.rename(columns={'goalsFor': 'goals'})
    if 'goalsAgainstFor' in df.columns and 'goalsAgainst' not in df.columns:
        df = df.rename(columns={'goalsAgainstFor': 'goalsAgainst'})
    if 'xGoalsFor' in df.columns and 'xGoals' not in df.columns:
        df = df.rename(columns={'xGoalsFor': 'xGoals'})
    if 'xGoalsForPercent' in df.columns and 'xGoalsPercentage' not in df.columns:
        df = df.rename(columns={'xGoalsForPercent': 'xGoalsPercentage'})
    if 'corsiForPercent' in df.columns and 'corsiPercentage' not in df.columns:
        df = df.rename(columns={'corsiForPercent': 'corsiPercentage'})
    if 'highDangerShotsFor' in df.columns and 'highDangerShots' not in df.columns:
        df = df.rename(columns={'highDangerShotsFor': 'highDangerShots'})

    df['win'] = (df['goals'] > df['goalsAgainst']).astype(int)
    df['pts'] = df['win'] * 2
    return df


gl = None
try:
    gl = load_gamelog()
except Exception as e:
    st.error(f"Error loading data: {type(e).__name__}: {e}")

if gl is None:
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 NHL Trends")
    st.caption(f"Season {SEASON} | {gl['gameDate'].dt.date.max()} latest")
    st.divider()

    window = st.selectbox(
        "Rolling window (games)",
        options=[5, 10, 20, 0],
        format_func=lambda x: "Full Season" if x == 0 else f"Last {x}",
        index=1,
    )

    situation_note = st.empty()
    st.divider()

    if st.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption(f"Total game rows: {len(gl):,}")
    st.caption(f"Teams: {gl['team'].nunique()}")
    st.caption(f"Date range: {gl['gameDate'].dt.date.min()} → {gl['gameDate'].dt.date.max()}")


# ── Helper: get last-N rows per team ────────────────────────────────────────

def last_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return last n games per team (or all if n == 0)."""
    if n == 0:
        return df
    return df.groupby('team', group_keys=False).apply(
        lambda g: g.nlargest(n, 'gameDate')
    )


def streak(series: pd.Series) -> str:
    """Current W/L streak from a boolean win series (ordered oldest→newest)."""
    results = series.tolist()
    if not results:
        return '—'
    val = results[-1]
    count = 0
    for r in reversed(results):
        if r == val:
            count += 1
        else:
            break
    return f"W{count}" if val else f"L{count}"


# ── Build summary table ──────────────────────────────────────────────────────

def build_summary(df: pd.DataFrame, n: int) -> pd.DataFrame:
    sub = last_n(df, n)
    rows = []
    for team, g in sub.groupby('team'):
        g = g.sort_values('gameDate')
        gp  = len(g)
        w   = g['win'].sum()
        l   = gp - w
        gf  = g['goals'].mean()
        ga  = g['goalsAgainst'].mean()
        xgf = g['xGoals'].mean()        if 'xGoals'           in g.columns else np.nan
        xga = g['xGoalsAgainst'].mean() if 'xGoalsAgainst'    in g.columns else np.nan
        xgp = g['xGoalsPercentage'].mean() * 100 if 'xGoalsPercentage' in g.columns else (
              xgf / (xgf + xga) * 100 if (xgf + xga) > 0 else np.nan
        )
        cf  = g['corsiPercentage'].mean() * 100 if 'corsiPercentage' in g.columns else np.nan
        hdp = (g['highDangerShots'].mean() /
               (g['highDangerShots'].mean() + g['highDangerShotsAgainst'].mean()) * 100
               if ('highDangerShots' in g.columns and 'highDangerShotsAgainst' in g.columns)
                  and (g['highDangerShots'].mean() + g['highDangerShotsAgainst'].mean()) > 0
               else np.nan)
        pts_pct = w / gp * 100 if gp else 0
        cur_streak = streak(g['win'])

        rows.append({
            'Team':      team,
            'GP':        gp,
            'W':         int(w),
            'L':         int(l),
            'W%':        round(pts_pct, 1),
            'GF/G':      round(gf, 2),
            'GA/G':      round(ga, 2),
            'GF−GA':     round(gf - ga, 2),
            'xGF/G':     round(xgf, 2) if not np.isnan(xgf) else None,
            'xGA/G':     round(xga, 2) if not np.isnan(xga) else None,
            'xGF%':      round(xgp, 1) if not np.isnan(xgp) else None,
            'CF%':       round(cf, 1)  if not np.isnan(cf)  else None,
            'HD%':       round(hdp, 1) if not np.isnan(hdp) else None,
            'Streak':    cur_streak,
        })

    return pd.DataFrame(rows).sort_values('W%', ascending=False).reset_index(drop=True)


# ── Build luck vs skill table ────────────────────────────────────────────────

def build_luck(df: pd.DataFrame) -> pd.DataFrame:
    """Season-long: actual W% vs xGF% — teams above the line are 'lucky'."""
    rows = []
    for team, g in df.groupby('team'):
        gp  = len(g)
        w_pct = g['win'].mean() * 100
        xgf = g['xGoals'].mean()        if 'xGoals'           in g.columns else np.nan
        xga = g['xGoalsAgainst'].mean() if 'xGoalsAgainst'    in g.columns else np.nan
        xgf_pct = xgf / (xgf + xga) * 100 if (not np.isnan(xgf) and (xgf + xga) > 0) else np.nan
        gf  = g['goals'].mean()
        ga  = g['goalsAgainst'].mean()
        pyth = gf**2 / (gf**2 + ga**2) * 100 if (gf + ga) > 0 else np.nan
        rows.append({
            'Team':       team,
            'GP':         gp,
            'W%':         round(w_pct, 1),
            'xGF%':       round(xgf_pct, 1) if not np.isnan(xgf_pct) else None,
            'Pyth%':      round(pyth, 1) if not np.isnan(pyth) else None,
            'Luck (W%−xGF%)':   round(w_pct - xgf_pct, 1) if not np.isnan(xgf_pct) else None,
            'Luck (W%−Pyth%)':  round(w_pct - pyth, 1) if not np.isnan(pyth) else None,
        })
    df_out = pd.DataFrame(rows)
    df_out['_sort'] = df_out['Luck (W%−xGF%)'].fillna(0)
    return df_out.sort_values('_sort', ascending=False).drop(columns=['_sort']).reset_index(drop=True)


# ── Main ─────────────────────────────────────────────────────────────────────

st.title("NHL Trending Teams")

label = "Full Season" if window == 0 else f"Last {window} Games"
tab_form, tab_luck, tab_team = st.tabs([
    f"Form ({label})",
    "Luck vs Skill (Season)",
    "Team Drill-Down",
])

# ── Tab 1: Form table ────────────────────────────────────────────────────────

with tab_form:
    summary = build_summary(gl, window)

    def color_pct(val):
        if pd.isna(val):
            return ''
        if val >= 55:   return 'background-color:#1a7a3a; color:white'
        if val >= 52:   return 'background-color:#c6efce; color:#276221'
        if val >= 48:   return ''
        if val >= 45:   return 'background-color:#fce4e4; color:#9c1c1c'
        return 'background-color:#f4b8b8; color:#7a0000'

    def color_diff(val):
        if pd.isna(val):
            return ''
        if val >  0.4: return 'background-color:#1a7a3a; color:white'
        if val >  0.1: return 'background-color:#c6efce; color:#276221'
        if val > -0.1: return ''
        if val > -0.4: return 'background-color:#fce4e4; color:#9c1c1c'
        return 'background-color:#f4b8b8; color:#7a0000'

    styled = (
        summary.style
        .applymap(color_pct,  subset=['W%', 'xGF%', 'CF%', 'HD%'])
        .applymap(color_diff, subset=['GF−GA'])
    )
    st.dataframe(styled, hide_index=True, use_container_width=True, height=1100)

# ── Tab 2: Luck vs Skill ─────────────────────────────────────────────────────

with tab_luck:
    luck = build_luck(gl)

    c1, c2 = st.columns([3, 2])

    with c1:
        st.subheader("W% vs xGF% (Season)")
        plot_df = luck.dropna(subset=['xGF%', 'W%'])
        fig = px.scatter(
            plot_df,
            x='xGF%', y='W%',
            text='Team',
            color='Luck (W%−xGF%)',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            range_color=[-15, 15],
            labels={'xGF%': 'xGF% (process)', 'W%': 'Win % (results)'},
        )
        # 45-degree reference line
        lo = plot_df[['xGF%', 'W%']].min().min() - 2
        hi = plot_df[['xGF%', 'W%']].max().max() + 2
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi],
            mode='lines',
            line=dict(color='grey', dash='dash', width=1),
            showlegend=False,
        ))
        fig.update_traces(textposition='top center', marker_size=10)
        fig.update_layout(height=500, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Teams **above** the line win more than their xG suggests (getting 'lucky' or clutch). "
            "Teams **below** deserve better results."
        )

    with c2:
        st.subheader("Luck Rankings")

        def color_luck(val):
            if pd.isna(val): return ''
            if val >  5:  return 'background-color:#f4b8b8; color:#7a0000'
            if val >  2:  return 'background-color:#fce4e4; color:#9c1c1c'
            if val < -5:  return 'background-color:#1a7a3a; color:white'
            if val < -2:  return 'background-color:#c6efce; color:#276221'
            return ''

        display_luck = luck[['Team', 'GP', 'W%', 'xGF%', 'Luck (W%−xGF%)', 'Pyth%', 'Luck (W%−Pyth%)']].copy()
        styled_luck = display_luck.style.applymap(
            color_luck, subset=['Luck (W%−xGF%)', 'Luck (W%−Pyth%)']
        )
        st.dataframe(styled_luck, hide_index=True, use_container_width=True, height=1050)

# ── Tab 3: Team Drill-Down ───────────────────────────────────────────────────

with tab_team:
    teams_sorted = sorted(gl['team'].unique())
    selected_team = st.selectbox("Select Team", teams_sorted)

    tgl = gl[gl['team'] == selected_team].sort_values('gameDate').copy()
    tgl['game_num'] = range(1, len(tgl) + 1)

    roll = st.slider("Rolling window", min_value=3, max_value=20, value=10)

    # Rolling metrics
    if 'xGoalsPercentage' in tgl.columns:
        tgl['roll_xGF%'] = tgl['xGoalsPercentage'].rolling(roll).mean() * 100
    elif 'xGoals' in tgl.columns and 'xGoalsAgainst' in tgl.columns:
        xgf_pct = tgl['xGoals'] / (tgl['xGoals'] + tgl['xGoalsAgainst'])
        tgl['roll_xGF%'] = xgf_pct.rolling(roll).mean() * 100

    if 'corsiPercentage' in tgl.columns:
        tgl['roll_CF%'] = tgl['corsiPercentage'].rolling(roll).mean() * 100

    tgl['roll_GF'] = tgl['goals'].rolling(roll).mean()
    tgl['roll_GA'] = tgl['goalsAgainst'].rolling(roll).mean()
    tgl['roll_W%'] = tgl['win'].rolling(roll).mean() * 100

    col_a, col_b, col_c, col_d = st.columns(4)
    recent = tgl.tail(window if window > 0 else len(tgl))
    col_a.metric("W%",   f"{recent['win'].mean()*100:.1f}%")
    col_b.metric("GF/G", f"{recent['goals'].mean():.2f}")
    col_c.metric("GA/G", f"{recent['goalsAgainst'].mean():.2f}")
    if 'xGoals' in tgl.columns:
        xgf = recent['xGoals'].mean()
        xga = recent['xGoalsAgainst'].mean()
        col_d.metric("xGF%", f"{xgf/(xgf+xga)*100:.1f}%" if (xgf + xga) > 0 else "—")

    # Chart: rolling xGF%, CF%, W%
    chart_cols = [c for c in ['roll_xGF%', 'roll_CF%', 'roll_W%'] if c in tgl.columns]
    if chart_cols:
        fig2 = go.Figure()
        colors = {'roll_xGF%': '#1f77b4', 'roll_CF%': '#ff7f0e', 'roll_W%': '#2ca02c'}
        labels = {'roll_xGF%': f'xGF% ({roll}G)', 'roll_CF%': f'CF% ({roll}G)', 'roll_W%': f'W% ({roll}G)'}
        for col in chart_cols:
            fig2.add_trace(go.Scatter(
                x=tgl['gameDate'], y=tgl[col],
                name=labels[col],
                line=dict(color=colors.get(col)),
                mode='lines',
            ))
        fig2.add_hline(y=50, line_dash='dash', line_color='grey', line_width=1)
        fig2.update_layout(height=350, yaxis_title='%', xaxis_title='Date',
                           legend=dict(orientation='h', y=1.1))
        st.plotly_chart(fig2, use_container_width=True)

    # GF/GA rolling chart
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=tgl['gameDate'], y=tgl['roll_GF'],
                              name=f'GF/G ({roll}G)', line=dict(color='green'), mode='lines'))
    fig3.add_trace(go.Scatter(x=tgl['gameDate'], y=tgl['roll_GA'],
                              name=f'GA/G ({roll}G)', line=dict(color='red'), mode='lines'))
    fig3.update_layout(height=280, yaxis_title='Goals/Game', xaxis_title='Date',
                       legend=dict(orientation='h', y=1.1))
    st.plotly_chart(fig3, use_container_width=True)

    # Recent games table
    st.subheader(f"Recent Games — {selected_team}")
    show_cols = ['gameDate', 'home_or_away', 'opposingTeam', 'goals', 'goalsAgainst', 'win']
    optional = ['xGoals', 'xGoalsAgainst', 'xGoalsPercentage', 'corsiPercentage',
                'highDangerShots', 'highDangerShotsAgainst']
    show_cols += [c for c in optional if c in tgl.columns]
    recent_display = tgl[show_cols].tail(20).sort_values('gameDate', ascending=False).copy()
    recent_display['gameDate'] = recent_display['gameDate'].dt.date
    if 'xGoalsPercentage' in recent_display.columns:
        recent_display['xGoalsPercentage'] = (recent_display['xGoalsPercentage'] * 100).round(1)
    if 'corsiPercentage' in recent_display.columns:
        recent_display['corsiPercentage'] = (recent_display['corsiPercentage'] * 100).round(1)
    recent_display = recent_display.rename(columns={
        'gameDate': 'Date', 'home_or_away': 'H/A', 'opposingTeam': 'Opp',
        'goals': 'GF', 'goalsAgainst': 'GA', 'win': 'W',
        'xGoals': 'xGF', 'xGoalsAgainst': 'xGA',
        'xGoalsPercentage': 'xGF%', 'corsiPercentage': 'CF%',
        'highDangerShots': 'HDSF', 'highDangerShotsAgainst': 'HDSA',
    })

    def color_win(val):
        if val == 1: return 'background-color:#c6efce; color:#276221'
        if val == 0: return 'background-color:#fce4e4; color:#9c1c1c'
        return ''

    st.dataframe(
        recent_display.style.applymap(color_win, subset=['W']),
        hide_index=True,
        use_container_width=True,
    )
