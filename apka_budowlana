# %%
# ProjectScope Construction v2.0 - Risk + Cost + ROI Predictor
# Dla: Technical Managerów i Architektów w Budownictwie
# pip install streamlit pandas scikit-learn plotly numpy

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="ProjectScope Construction", layout="wide", page_icon="🏗️")

st.title("🏗️ ProjectScope Construction v2.0")
st.markdown("**Przewiduj ryzyko, koszt i opłacalność etapów budowy na podstawie danych historycznych**")

# %%
# 1. GENERATOR DANYCH - REALIA BUDOWLANE
@st.cache_data
def generate_construction_data():
    np.random.seed(42)
    n = 600

    etapy = ['Fundamenty', 'Stan surowy otwarty', 'Stan surowy zamknięty', 'Dach',
             'Instalacje', 'Tynki/Posadzki', 'Elewacja', 'Wykończenie', 'Otoczenie']
    typy = ['Dom jednorodzinny', 'Bliźniak', 'Szeregowiec', 'Budynek wielorodzinny', 'Hala', 'Biuro']
    wykonawcy = ['Własna ekipa', 'Generalny wykonawca', 'Podwykonawca A', 'Podwykonawca B']
    pory_roku = ['Wiosna', 'Lato', 'Jesień', 'Zima']
    klasy_dzialki = ['Łatwa', 'Średnia', 'Trudna'] # ukształtowanie, dostęp

    data = {
        'etap': np.random.choice(etapy, n, p=[0.12, 0.15, 0.1, 0.08, 0.18, 0.12, 0.08, 0.12, 0.05]),
        'typ_budynku': np.random.choice(typy, n, p=[0.35, 0.15, 0.15, 0.2, 0.1, 0.05]),
        'wykonawca': np.random.choice(wykonawcy, n, p=[0.4, 0.3, 0.15, 0.15]),
        'pora_roku': np.random.choice(pory_roku, n),
        'pow_m2': np.random.randint(80, 2000, n),
        'kondygnacje': np.random.choice([1, 2, 3, 4, 5], n, p=[0.25, 0.45, 0.2, 0.08, 0.02]),
        'zmiany_inwestora': np.random.poisson(1.8, n), # ile razy zmieniał projekt
        'klasa_dzialki': np.random.choice(klasy_dzialki, n, p=[0.5, 0.3, 0.2]),
        'pozwolenia_czas': np.random.randint(30, 180, n), # dni czekania na papiery
    }
    df = pd.DataFrame(data)

    # SYMULACJA REALIÓW BUDOWY
    baza_dni = df['pow_m2'] * 0.12 + df['kondygnacje'] * 15

    kara_czas = (df['etap'] == 'Fundamenty') * (df['pora_roku'] == 'Zima') * 25 + \
                (df['etap'] == 'Dach') * (df['pora_roku'] == 'Zima') * 20 + \
                (df['etap'] == 'Elewacja') * (df['pora_roku'] == 'Zima') * 15 + \
                (df['wykonawca']!= 'Własna ekipa') * 10 + \
                df['zmiany_inwestora'] * 7 + \
                (df['klasa_dzialki'] == 'Trudna') * 15 + \
                (df['klasa_dzialki'] == 'Średnia') * 5 + \
                (df['pozwolenia_czas'] > 120) * 10

    df['actual_days'] = baza_dni + kara_czas + np.random.normal(0, 8, n)
    df['actual_days'] = df['actual_days'].clip(lower=7)
    df['estimate_days'] = baza_dni * 1.15
    df['overrun_pct'] = ((df['actual_days'] - df['estimate_days']) / df['estimate_days'] * 100).round(1)

    # BUDŻET I KOSZTY
    koszt_m2 = {'Dom jednorodzinny': 4500, 'Bliźniak': 4200, 'Szeregowiec': 4000,
                'Budynek wielorodzinny': 5500, 'Hala': 2800, 'Biuro': 6500}
    baza_koszt = df['pow_m2'] * df['typ_budynku'].map(koszt_m2)

    kara_koszt = kara_czas * 1200 + \
                 df['zmiany_inwestora'] * 25000 + \
                 (df['klasa_dzialki'] == 'Trudna') * df['pow_m2'] * 300 + \
                 (df['pora_roku'] == 'Zima') * df['pow_m2'] * 150

    df['actual_cost_pln'] = baza_koszt + kara_koszt + np.random.normal(0, 50000, n)
    df['estimate_cost_pln'] = baza_koszt * 1.2
    df['budget_overrun_pct'] = ((df['actual_cost_pln'] - df['estimate_cost_pln']) / df['estimate_cost_pln'] * 100).round(1)

    # WARTOŚĆ INWESTYCJI / PRZYCHÓD ZE SPRZEDAŻY
    cena_sprzedazy_m2 = {'Dom jednorodzinny': 9000, 'Bliźniak': 8500, 'Szeregowiec': 8200,
                         'Budynek wielorodzinny': 11000, 'Hala': 4000, 'Biuro': 14000}
    df['revenue_pln'] = df['pow_m2'] * df['typ_budynku'].map(cena_sprzedazy_m2) * np.random.uniform(0.95, 1.05, n)
    df['profit_pln'] = df['revenue_pln'] - df['actual_cost_pln']
    df['roi_pct'] = (df['profit_pln'] / df['actual_cost_pln'] * 100).round(1)

    # FLAGI RYZYKA
    df['time_risk'] = (df['overrun_pct'] > 20).astype(int) # >20% obsuwy to ryzyko
    df['budget_risk'] = (df['budget_overrun_pct'] > 15).astype(int) # >15% budżetu to ryzyko
    df['low_roi'] = (df['roi_pct'] < 10).astype(int) # <10% ROI to słabo

    return df.round(2)

# %%
# 2. SIDEBAR I DANE
st.sidebar.header("1. Dane historyczne")
uploaded_file = st.sidebar.file_uploader("Wgraj CSV z historią budów", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Wczytano {len(df)} rekordów")
else:
    df = generate_construction_data()
    st.sidebar.info("Używasz danych demo - 600 etapów budowy")

# %%
# 3. TRENING 4 MODELI ML
with st.spinner('Uczę modele AI na danych z budowy...'):
    features = ['etap', 'typ_budynku', 'wykonawca', 'pora_roku', 'pow_m2',
                'kondygnacje', 'zmiany_inwestora', 'klasa_dzialki', 'pozwolenia_czas']
    X = df[features].copy()

    le_dict = {}
    for col in ['etap', 'typ_budynku', 'wykonawca', 'pora_roku', 'klasa_dzialki']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        le_dict[col] = le

    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    # Model 1: Ryzyko obsuwy czasu
    clf_time = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_time.fit(X_train, df.loc[X_train.index, 'time_risk'])

    # Model 2: Ryzyko przekroczenia budżetu
    clf_budget = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_budget.fit(X_train, df.loc[X_train.index, 'budget_risk'])

    # Model 3: O ile % przekroczymy czas
    reg_time = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_time.fit(X_train, df.loc[X_train.index, 'overrun_pct'])

    # Model 4: Przewidywany koszt końcowy
    reg_cost = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_cost.fit(X_train, df.loc[X_train.index, 'actual_cost_pln'])

st.sidebar.header("2. Skuteczność AI")
st.sidebar.metric("Predykcja obsuwy", f"{clf_time.score(X_test, df.loc[X_test.index, 'time_risk'])*100:.1f}%")
st.sidebar.metric("Predykcja budżetu", f"{clf_budget.score(X_test, df.loc[X_test.index, 'budget_risk'])*100:.1f}%")

# %%
# 4. DASHBOARD HISTORYCZNY
st.header("📊 Benchmark Historyczny")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Śr. obsuwa czasu", f"{df['overrun_pct'].mean():.1f}%")
c2.metric("Śr. przekr. budżetu", f"{df['budget_overrun_pct'].mean():.1f}%")
c3.metric("Śr. ROI projektu", f"{df['roi_pct'].mean():.1f}%")
c4.metric("Śr. zysk z m²", f"{(df['profit_pln'] / df['pow_m2']).mean():,.0f} zł")

tab1, tab2, tab3 = st.tabs(["⏱️ Czas", "💰 Budżet", "📈 ROI"])

with tab1:
    fig = px.box(df, x='etap', y='overrun_pct', color='pora_roku',
                 title='Rozkład obsuw czasowych wg etapu budowy')
    fig.add_hline(y=0, line_dash="dash", line_color="green", annotation_text="Zgodnie z planem")
    fig.add_hline(y=20, line_dash="dash", line_color="red", annotation_text="Próg ryzyka")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = px.scatter(df, x='pow_m2', y='actual_cost_pln', color='typ_budynku',
                     size='zmiany_inwestora', hover_data=['etap', 'wykonawca'],
                     title='Koszt rzeczywisty vs powierzchnia', trendline="ols")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    fig = px.bar(df.groupby('typ_budynku')['roi_pct'].mean().reset_index().sort_values('roi_pct'),
                 x='roi_pct', y='typ_budynku', orientation='h',
                 title='Średni ROI wg typu budynku', text='roi_pct')
    st.plotly_chart(fig, use_container_width=True)

# %%
# 5. KALKULATOR PREDYKCYJNY
st.header("🔮 Kalkulator Ryzyka i Opłacalności Nowego Etapu")

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Parametry budowy")
    pred_etap = st.selectbox("Etap budowy", df['etap'].unique())
    pred_typ = st.selectbox("Typ budynku", df['typ_budynku'].unique())
    pred_wykonawca = st.selectbox("Wykonawca", df['wykonawca'].unique())

with col2:
    st.subheader("Warunki")
    pred_pora = st.selectbox("Start robót", df['pora_roku'].unique())
    pred_pow = st.slider("Powierzchnia m²", 50, 3000, 200)
    pred_kond = st.slider("Kondygnacje", 1, 6, 2)
    pred_dzialka = st.selectbox("Klasa działki", df['klasa_dzialki'].unique())

with col3:
    st.subheader("Estymaty i ryzyka")
    pred_zmiany = st.slider("Zakładane zmiany inwestora", 0, 15, 2)
    pred_pozwolenia = st.slider("Czas na pozwolenia [dni]", 30, 200, 60)
    pred_est_days = st.number_input("Estymata czas [dni]", value=45)
    pred_est_cost = st.number_input("Estymata koszt [PLN]", value=pred_pow*4500)

# PREDYKCJA - ODPORNA NA INDEX ERROR
pred_data = pd.DataFrame([{
    'etap': le_dict['etap'].transform([pred_etap])[0],
    'typ_budynku': le_dict['typ_budynku'].transform([pred_typ])[0],
    'wykonawca': le_dict['wykonawca'].transform([pred_wykonawca])[0],
    'pora_roku': le_dict['pora_roku'].transform([pred_pora])[0],
    'pow_m2': pred_pow,
    'kondygnacje': pred_kond,
    'zmiany_inwestora': pred_zmiany,
    'klasa_dzialki': le_dict['klasa_dzialki'].transform([pred_dzialka])[0],
    'pozwolenia_czas': pred_pozwolenia
}])

# Bezpieczne wywołanie predict_proba
def safe_proba(clf, data):
    proba = clf.predict_proba(data)[0]
    return proba[1] * 100 if len(proba) > 1 else (100 if clf.predict(data)[0] == 1 else 0)

risk_time = safe_proba(clf_time, pred_data)
risk_budget = safe_proba(clf_budget, pred_data)
overrun_pred = reg_time.predict(pred_data)[0]
cost_pred = reg_cost.predict(pred_data)[0]
revenue_est = pred_pow * 9000 # uproszczona estymata przychodu
profit_pred = revenue_est - cost_pred
roi_pred = (profit_pred / cost_pred * 100) if cost_pred > 0 else 0

st.divider()
st.subheader("🎯 Wynik Analizy AI dla Piotrka")

r1, r2, r3, r4 = st.columns(4)

with r1:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=risk_time, title={'text': "Ryzyko obsuwy >20%"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "red" if risk_time > 70 else "orange" if risk_time > 40 else "green"},
               'steps': [{'range': [0, 40], 'color': "lightgreen"},
                         {'range': [40, 70], 'color': "yellow"}]}))
    st.plotly_chart(fig, use_container_width=True)

with r2:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=risk_budget, title={'text': "Ryzyko przekr. budżetu >15%"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "red" if risk_budget > 70 else "orange" if risk_budget > 40 else "green"},
               'steps': [{'range': [0, 40], 'color': "lightgreen"},
                         {'range': [40, 70], 'color': "yellow"}]}))
    st.plotly_chart(fig, use_container_width=True)

with r3:
    st.metric("Przewidywany czas", f"{pred_est_days * (1 + overrun_pred/100):.0f} dni",
              delta=f"{overrun_pred:+.1f}% vs estymata")
    st.metric("Przewidywany koszt", f"{cost_pred:,.0f} zł",
              delta=f"{(cost_pred - pred_est_cost):+,.0f} zł vs estymata")

with r4:
    st.metric("Szacowany przychód", f"{revenue_est:,.0f} zł")
    st.metric("Przewidywany zysk", f"{profit_pred:,.0f} zł")
    st.metric("ROI projektu", f"{roi_pred:.1f}%")

# REKOMENDACJE DLA TECH MANAGERA
st.subheader("💡 Rekomendacje Zarządcze")
if risk_time > 70 or risk_budget > 70:
    st.error("""
    **🚨 WYSOKIE RYZYKO - DZIAŁAJ:**
    1. Zabezpiecz +25% buforu czasowego i +20% budżetowego w umowie
    2. Rozważ zmianę terminu startu - zima = dramat na fundamentach/dachu
    3. Twarda negocjacja z inwestorem: każda zmiana = aneks + dopłata
    4. Codzienny nadzór własnej ekipy zamiast podwykonawcy
    """)
elif risk_time > 40 or risk_budget > 40:
    st.warning("""
    **⚠️ ŚREDNIE RYZYKO - ZABEZPIECZ SIĘ:**
    1. Dodaj 15% buforu czasowego i 10% budżetowego
    2. Zapisz w umowie kary za zmiany inwestora po starcie robót
    3. Zrób szczegółową dokumentację zdjęciową przed każdym etapem
    """)
else:
    st.success("""
    **✅ NISKIE RYZYKO - WARUNKI OPTYMALNE:**
    1. Harmonogram i budżet wyglądają realnie
    2. Pilnuj żeby inwestor nie dokładał zmian w trakcie
    3. To jest projekt na którym zarobisz. Dawaj w palnik!
    """)

if roi_pred < 10:
    st.info("💸 **UWAGA ROI:** Przy tym ROI <10% zastanów się czy projekt ma sens ekonomiczny. W budowlance standard to 15-25%.")

# %%
# 6. CO WPŁYWA NA RYZYKO
st.header("🧠 TOP Czynniki Ryzyka na Budowie")
importance = pd.DataFrame({
    'feature': features,
    'importance': clf_time.feature_importances_
}).sort_values('importance', ascending=True)

fig = px.bar(importance, x='importance', y='feature', orientation='h',
             title='Co najbardziej rozwala harmonogram?')
st.plotly_chart(fig, use_container_width=True)

st.caption("ProjectScope Construction v2.0 | Built with Meta AI for Polish Construction Tech Managers 🏗️")
st.caption("Dane demo. Podłącz CSV z Waszych budów żeby trenować na realnych projektach.")
