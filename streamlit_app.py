# streamlit_app.py
import os
import pandas as pd
import streamlit as st
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col, lit
import jwt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import json
import joblib
import shap
import numpy as np 


# ---------------------------------------------------------------
# 1) Load secrets once
# ---------------------------------------------------------------
SF  = st.secrets["snowflake"]
APP = st.secrets["app"]

JWT_SECRET   = APP.get("jwt_secret", "CHANGE_ME_TO_A_LONG_RANDOM_SECRET")
ADMIN_TOKEN  = APP.get("admin_token", "")
STD_TABLE    = APP.get("std_table", "ETL_DB.ETL_SCH.STD")
RAW_TABLE    = APP.get("raw_table", "ETL_DB.ETL_SCH.RAW")
STAGE_NAME   = APP.get("stage_name", "PAC_STAGE")
APP_BASE_URL = APP.get("app_base_url", "http://localhost:8501")

st.set_page_config(page_title="🏥 ProActive Care Unit", layout="wide")
st.title("🏥 ProActive Care Unit")

# ---------------------------------------------------------------
# 2) Session factory (single source of truth)
# ---------------------------------------------------------------
def get_session() -> Session:
    return Session.builder.configs({
        "account":   SF["account"],
        "user":      SF["user"],
        "password":  SF["password"],
        "role":      SF.get("role", ""),
        "warehouse": SF.get("warehouse", ""),
        "database":  SF.get("database", ""),
        "schema":    SF.get("schema", "")
    }).create()

# Create one session
session = get_session()

# ---------------------------------------------------------------
# 3) Auth helpers
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def validate_token(token: str):
    if not token:
        return None, "Missing token"
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        if payload.get("scope") != "patient_dashboard":
            return None, "Invalid scope"
        sub = payload.get("sub")
        if not sub:
            return None, "Token missing patient id (sub)"
        return str(sub), None
    except jwt.ExpiredSignatureError:
        return None, "Token expired"
    except jwt.InvalidTokenError:
        return None, "Invalid token"

# ---------------------------------------------------------------
# 4) Data access (Snowpark-safe)
# ---------------------------------------------------------------
def fetch_latest_prediction(session: Session, patient_id: str) -> pd.DataFrame:
    """
    Uses parameter binding (?) instead of %s.
    QUALIFY with ROW_NUMBER gets the most recent active row per patient.
    """
    q = f"""
        SELECT *
        FROM {RAW_TABLE}
        WHERE PATIENT_ID = ?
    """
    return session.sql(q, params=[patient_id]).to_pandas()

def fetch_recent_vitals(session: Session, patient_id: str, limit: int = 50) -> pd.DataFrame:
    """
    Snowpark DataFrame API (no raw SQL needed for values).
    Keeps LIMIT safe and typed.
    """
    cols = [
        "ENCOUNTER_ID","PATIENT_ID","AGE","BMI","WEIGHT","HEIGHT",
        "HEART_RATE_APACHE","MAP_APACHE","RESPRATE_APACHE","TEMP_APACHE",
        "INTUBATED_APACHE","VENTILATED_APACHE","ICU_ADMIT_SOURCE","ICU_STAY_TYPE",
        "APACHE_2_DIAGNOSIS","APACHE_3J_DIAGNOSIS","GENDER","HOSPITAL_DEATH","INSERT_TIMESTMP"
    ]
    df = (
        session.table(RAW_TABLE)
        .filter(col("PATIENT_ID") == lit(patient_id))
        .select(*[c for c in cols if c in session.table(RAW_TABLE).schema.names])
        .sort(col("INSERT_TIMESTMP").desc())
        .limit(int(limit))
    )
    return df.to_pandas()

# ---------------------------------------------------------------
# 5) UI
# ---------------------------------------------------------------
tab_patient, tab_admin = st.tabs(["🔑 Patient", "🛠️ Admin"])

# ---------------- Patient tab ----------------
with tab_patient:
    # Get token from URL (?token=...) if present, else empty
    qp_token = st.query_params.get("token")
    if isinstance(qp_token, list):
        qp_token = qp_token[0]

    token = st.text_input("Paste your secure access token", value=qp_token or "", type="password")
    if not token:
        st.info("Missing token. Use your emailed link (it includes ?token=...) or paste the token above.")
        #st.stop()

    patient_id, err = validate_token(token)
    if err:
        st.error(f"Access denied: {err}")
        #st.stop()

    st.success(f"Access granted for Patient ID: **{patient_id}**")

    with st.spinner("Loading your latest risk evaluation..."):
        pred = fetch_latest_prediction(session, patient_id)

    if pred.empty:
        st.warning("No recent prediction found yet. Please check back later.")
        #st.stop()

    row = pred.iloc[0]
    pred_val = int(row.get("HOSPITAL_DEATH", 0))
    label = "High Risk" if pred_val == 1 else "Lower Risk"

    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction", label)
    c2.metric("Encounter ID", str(row.get("ENCOUNTER_ID", "")))
    c3.metric("Last Updated (UTC)", str(row.get("INSERT_TIMESTMP", "")))

    st.divider()
    st.markdown("### Recent Vitals")
    vitals = fetch_recent_vitals(session, patient_id, limit=50)
    if vitals.empty:
        st.info("No vitals available.")
    else:
        keep = [c for c in [
            "ENCOUNTER_ID","AGE","BMI","WEIGHT","HEIGHT",
            "HEART_RATE_APACHE","MAP_APACHE","RESPRATE_APACHE","TEMP_APACHE",
            "INTUBATED_APACHE","VENTILATED_APACHE","ICU_ADMIT_SOURCE","ICU_STAY_TYPE",
            "APACHE_2_DIAGNOSIS","APACHE_3J_DIAGNOSIS","GENDER","INSERT_TIMESTMP"
        ] if c in vitals.columns]
        st.dataframe(vitals[keep], use_container_width=True)
        # ======================= SNAPSHOT VISUALS (single-encounter friendly) =======================

        # 1) Gauge / Dial — overall risk (uses probability column if present, else a fallback based on label)
        st.markdown("### 🎯 Overall Risk")
        risk_prob = None
        try:
            # try common column names for probability (adjust if your STD has a specific column)
            for k in ["RISK_PROB","P_HOSPITAL_DEATH","HOSPITAL_DEATH_PROB","PRED_PROB","PROB"]:
                if k in pred.columns and pd.notna(row.get(k)):
                    risk_prob = float(row.get(k))
                    break
        except Exception:
            pass

        if risk_prob is None:
            # fallback if you only have a 0/1 label
            risk_prob = 0.85 if pred_val == 1 else 0.15

        fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_prob * 100,
        number={'suffix': "%"},
        title={'text': "Mortality Risk"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "crimson" if risk_prob >= 0.5 else "seagreen"},
            'steps': [
                {'range': [0, 50],  'color': "#cfeecf"},
                {'range': [50, 75], 'color': "#fff3b0"},
                {'range': [75, 100],'color': "#ffc9c9"},
            ]
        }
    ))
    st.plotly_chart(fig_g, use_container_width=True)


    # 2) Radar (Spider) — patient’s vitals vs reference values
    st.markdown("### 🧭 Vitals Profile (vs Reference)")
    vital_features = [c for c in ["AGE","BMI","HEART_RATE_APACHE","MAP_APACHE","RESPRATE_APACHE","TEMP_APACHE"] if c in vitals.columns]
    if len(vital_features) >= 3:
        patient_vals = vitals.iloc[0][vital_features].astype(float).tolist()
        # reference: replace with your cohort means if you have them
        reference_defaults = {
         "AGE": 60, "BMI": 26, "HEART_RATE_APACHE": 85, "MAP_APACHE": 90,
         "RESPRATE_APACHE": 18, "TEMP_APACHE": 37
        }
        ref_vals = [float(reference_defaults.get(f, np.nan)) for f in vital_features]

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(r=patient_vals, theta=vital_features, fill='toself', name='Patient'))
        fig_r.add_trace(go.Scatterpolar(r=ref_vals,     theta=vital_features, fill='toself', name='Reference'))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
        st.plotly_chart(fig_r, use_container_width=True)
    else:
        st.caption("Not enough numeric vitals to draw a radar chart.")


    # 3) SHAP — top feature contributions for THIS patient (optional: requires saved model artifacts)
    st.markdown("### 🔍 Why this prediction? (SHAP)")
    try:
        import shap
        # Expect training artifacts saved by your training script:
        #   models/rf_model.joblib
        #   models/feature_columns.json  (columns used during training after one-hot)
        rf_model = joblib.load("models/rf_model.joblib")
        with open("models/feature_columns.json", "r", encoding="utf-8") as f:
            feature_cols = json.load(f)

        # build single-row features from vitals (newest row already in scope)
        x_row = vitals[keep].head(1).copy()
        # drop non-features (align with your training code)
        DROP_COLS = ["HOSPITAL_DEATH","ENCOUNTER_ID","PATIENT_ID","HOSPITAL_ID","INSERT_TIMESTMP"]
        x_row = x_row.drop(columns=[c for c in DROP_COLS if c in x_row.columns], errors="ignore")

        # one-hot + align to training matrix
        x_proc = pd.get_dummies(x_row, drop_first=True)
        x_proc = x_proc.reindex(columns=feature_cols, fill_value=0)

        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(x_proc)

        vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        contrib = pd.Series(vals, index=x_proc.columns).abs().sort_values(ascending=False).head(12)

        fig, ax = plt.subplots()
        contrib.sort_values().plot.barh(ax=ax, color="#ff8a80")
        ax.set_title("Top Feature Contributions (This Patient)")
        ax.set_xlabel("Absolute SHAP value (impact on risk)")
        st.pyplot(fig, clear_figure=True)

    except FileNotFoundError:
        st.info("SHAP not shown: training artifacts not found. Save model to `models/rf_model.joblib` and columns to `models/feature_columns.json`.")
    except ModuleNotFoundError as e:
        st.info(f"SHAP not shown: missing package ({e}). Install `shap` and `joblib`.")
    except Exception as e:
        st.warning(f"SHAP could not be rendered: {e}")


    # 4) Population Comparison — where this patient sits vs cohort (optional)
    st.markdown("### 📈 Where does this patient sit vs others?")
    pop_cols = [c for c in ["BMI","AGE","MAP_APACHE","HEART_RATE_APACHE"] if c in vitals.columns]
    for colname in pop_cols:
        try:
            # sample a cohort from RAW (avoid huge pulls)
            cohort = (
                session.table(RAW_TABLE)
                .select(colname)
                .dropna()
                .limit(5000)
                .to_pandas()[colname]
            )
            if cohort.empty or pd.isna(vitals.iloc[0][colname]):
                continue

            fig_d = ff.create_distplot([cohort.values], [colname], show_hist=False)
            # mark this patient's value
            fig_d.add_vline(
                x=float(vitals.iloc[0][colname]),
                line_dash="dash",
                line_color="red",
                annotation_text="This patient",
                annotation_position="top left"
            )
            fig_d.update_layout(title=f"{colname.replace('_',' ').title()} — Cohort Distribution")
            st.plotly_chart(fig_d, use_container_width=True)
        except Exception:
            # keep the UI clean if the column isn't present in RAW or any other issue
            pass

    # ===================== end SNAPSHOT VISUALS =====================


    st.caption("If you have concerns, contact your provider. In emergencies, call your local emergency number.")

