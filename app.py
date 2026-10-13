import streamlit as st

from data import LEAGUES, load_league
from model import MatchModel

st.set_page_config(page_title="Match Predictor", layout="centered")
st.title("Match Predictor")
st.caption("Poisson model trained on recent results. Probabilities, not guarantees.")


@st.cache_resource(ttl=6 * 3600, show_spinner="Loading results and fitting model...")
def get_model(league):
    df = load_league(league)
    if df.empty:
        return None, "Could not download results for this league."
    try:
        return MatchModel(df), None
    except ValueError as e:
        return None, str(e)


league = st.selectbox("League", list(LEAGUES))
model, error = get_model(league)
if error:
    st.error(error)
    st.stop()

st.caption(f"{model.n_matches} matches, latest {model.last_date:%d %b %Y}")

c1, c2 = st.columns(2)
home = c1.selectbox("Home team", model.teams, index=0)
away = c2.selectbox("Away team", model.teams, index=min(1, len(model.teams) - 1))

if home == away:
    st.warning("Pick two different teams.")
    st.stop()

p = model.predict(home, away)

st.subheader(f"{home} vs {away}")
m1, m2, m3 = st.columns(3)
m1.metric("Home win", f"{p['home']:.0%}", f"fair odds {1 / p['home']:.2f}", delta_color="off")
m2.metric("Draw", f"{p['draw']:.0%}", f"fair odds {1 / p['draw']:.2f}", delta_color="off")
m3.metric("Away win", f"{p['away']:.0%}", f"fair odds {1 / p['away']:.2f}", delta_color="off")

st.write(
    f"Expected goals: **{p['xg_home']:.2f}** - **{p['xg_away']:.2f}**. "
    f"Over 2.5 goals: **{p['over_2_5']:.0%}**. Both teams score: **{p['btts']:.0%}**."
)
st.write("Most likely scores: " + ", ".join(f"{h}-{a} ({pr:.0%})" for h, a, pr in p["scorelines"]))

st.subheader("Compare with bookmaker odds")
st.caption("Enter the odds you see. A positive edge means the model rates the outcome higher than the odds imply.")
o1, o2, o3 = st.columns(3)
odds = {
    "Home": o1.number_input("Home odds", min_value=1.0, value=2.0, step=0.05),
    "Draw": o2.number_input("Draw odds", min_value=1.0, value=3.3, step=0.05),
    "Away": o3.number_input("Away odds", min_value=1.0, value=3.5, step=0.05),
}
probs = {"Home": p["home"], "Draw": p["draw"], "Away": p["away"]}
rows = [{"Outcome": k, "Model": f"{probs[k]:.1%}", "Odds imply": f"{1 / v:.1%}",
         "Edge": f"{probs[k] - 1 / v:+.1%}"} for k, v in odds.items()]
st.table(rows)

st.caption("Not betting advice. Betting involves risk; only stake what you can afford to lose.")
