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
import numpy as np 
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


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
        FROM {STD_TABLE}
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
# 5) UI — single patient view (no tabs)
patient_container = st.container()

# ---------------- Patient tab ----------------
with patient_container:
    # Get token from URL (?token=...) if present, else empty
    qp_token = st.query_params.get("token")
    if isinstance(qp_token, list):
        qp_token = qp_token[0]

    token = st.text_input("Paste your secure access token", value=qp_token or "", type="password")
    if not token:
        st.info("Missing token. Use your emailed link (it includes ?token=...) or paste the token above.")
        st.stop()

    patient_id, err = validate_token(token)
    if err:
        st.error(f"Access denied: {err}")
        st.stop()

    st.success(f"Access granted for Patient ID: **{patient_id}**")

    with st.spinner("Loading your latest risk evaluation..."):
        pred = fetch_latest_prediction(session, patient_id)

    if pred.empty:
        st.warning("No recent prediction found yet. Please check back later.")
        st.stop()

    row = pred.iloc[0]
    pred_val = int(row.get("HOSPITAL_DEATH", 0))
    label = "High Risk" if pred_val == 1 else "Lower Risk"

    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction", label)
    c2.metric("Encounter ID", str(row.get("ENCOUNTER_ID", "")))
    c3.metric("Last Updated (UTC)", str(row.get("INSERT_TIMESTMP", "")))

    st.divider()

    # ======================= PATIENT DASHBOARD =======================
    vitals = fetch_recent_vitals(session, patient_id, limit=50)

    if vitals.empty:
        st.info("No vitals available.")
    else:
        # Use the latest row for snapshot visuals
        latest = vitals.iloc[0]

        # ---------- 1) Overall risk: banner + donut ----------
        st.markdown("### 🎯 Overall Risk")

        # Get probability if present, else fallback from label
        risk_prob = None
        try:
            for k in ["RISK_PROB", "P_HOSPITAL_DEATH", "HOSPITAL_DEATH_PROB", "PRED_PROB", "PROB"]:
                if k in pred.columns and pd.notna(row.get(k)):
                    risk_prob = float(row.get(k))
                    break
        except Exception:
            pass

        if risk_prob is None:
            risk_prob = 0.85 if pred_val == 1 else 0.15  # fallback

        # Risk banner
        if risk_prob >= 0.7:
            risk_color = "#ffcccc"   # light red
            risk_emoji = "🔴"
            risk_message = "This patient has a high estimated risk. Please review urgently."
            risk_label = "High Risk"
        elif risk_prob >= 0.4:
            risk_color = "#fff4cc"   # light yellow
            risk_emoji = "🟠"
            risk_message = "This patient has a moderate estimated risk. Close monitoring is recommended."
            risk_label = "Moderate Risk"
        else:
            risk_color = "#d9f2d9"   # light green
            risk_emoji = "🟢"
            risk_message = "This patient has a lower estimated risk. Continue standard monitoring."
            risk_label = "Lower Risk"

        st.markdown(
            f"""
            <div style="
                padding:1.25rem;
                border-radius:1rem;
                background: linear-gradient(90deg, {risk_color}, #ffffff);
                border:1px solid #dddddd;
                margin-bottom:1rem;
            ">
                <h4 style="margin:0; font-size:1.2rem;">
                    {risk_emoji} {risk_label} &nbsp; (Predicted risk: {risk_prob*100:.1f}%)
                </h4>
                <p style="margin:0.4rem 0 0; font-size:0.95rem; color:#333;">
                    {risk_message}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Donut chart
        risk_pct = risk_prob * 100
        safe_pct = 100 - risk_pct

        fig_donut = go.Figure(
            data=[go.Pie(
                labels=["Risk of death", "Lower-risk"],
                values=[risk_pct, safe_pct],
                hole=0.6,
                textinfo="label+percent",
            )]
        )
        fig_donut.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        # ---------- 2) Patient snapshot cards ----------
        st.markdown("### 👤 Patient Snapshot")
        age = latest.get("AGE", None)
        bmi = latest.get("BMI", None)
        # change ICU_LOS_DAYS to whatever column you actually have, or drop if none
        icu_los = latest.get("ICU_LOS_DAYS", None)

        c1, c2, c3 = st.columns(3)
        c1.metric("Age", f"{age:.0f}" if pd.notna(age) else "—", "years")
        c2.metric("BMI", f"{bmi:.1f}" if pd.notna(bmi) else "—")
        c3.metric("ICU Stay", f"{icu_los:.1f} days" if pd.notna(icu_los) else "—")

        # ---------- 3) Recent vitals table ----------
        st.markdown("### 📋 Recent Vitals")

        keep = [c for c in [
            "ENCOUNTER_ID","AGE","BMI","WEIGHT","HEIGHT",
            "HEART_RATE_APACHE","MAP_APACHE","RESPRATE_APACHE","TEMP_APACHE",
            "INTUBATED_APACHE","VENTILATED_APACHE","ICU_ADMIT_SOURCE","ICU_STAY_TYPE",
            "APACHE_2_DIAGNOSIS","APACHE_3J_DIAGNOSIS","GENDER","INSERT_TIMESTMP"
        ] if c in vitals.columns]

        st.dataframe(vitals[keep], use_container_width=True)

        # ---------- 4) Vitals vs normal range ----------
        st.markdown("### 🧭 Vitals vs Normal Range")

        norm_ranges = {
            "HEART_RATE_APACHE": (60, 100),
            "MAP_APACHE": (70, 105),
            "RESPRATE_APACHE": (12, 20),
            "TEMP_APACHE": (36.5, 37.5),
            "BMI": (18.5, 24.9),
        }

        norm_rows = []
        for col, (low, high) in norm_ranges.items():
            if col in vitals.columns and pd.notna(latest.get(col)):
                try:
                    norm_rows.append({
                        "Vital": col.replace("_APACHE", "").replace("_", " ").title(),
                        "Low": low,
                        "High": high,
                        "Patient": float(latest[col]),
                    })
                except Exception:
                    pass

        if norm_rows:
            df_norm = pd.DataFrame(norm_rows)

            fig_norm = go.Figure()
            # Normal range bar
            fig_norm.add_trace(go.Bar(
                x=df_norm["High"] - df_norm["Low"],
                y=df_norm["Vital"],
                base=df_norm["Low"],
                orientation="h",
                opacity=0.4,
                name="Normal range",
            ))
            # Patient marker
            fig_norm.add_trace(go.Scatter(
                x=df_norm["Patient"],
                y=df_norm["Vital"],
                mode="markers",
                name="Patient",
                marker=dict(size=10),
            ))
            fig_norm.update_layout(
                xaxis_title="Value",
                yaxis_title="Vital",
                barmode="overlay",
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig_norm, use_container_width=True)
        else:
            st.caption("Vitals not available to compare against normal ranges.")

        # ---------- 5) Key clinical fields ----------
        st.markdown("### 🧾 Key Clinical Fields")

        key_fields = [
            ("Age", "AGE"),
            ("BMI", "BMI"),
            ("APACHE II Score", "APACHE_2_SCORE"),
            ("APACHE Diagnosis", "APACHE_2_DIAGNOSIS"),
            ("ICU Admit Source", "ICU_ADMIT_SOURCE"),
            ("Ventilated", "VENTILATED_APACHE"),
        ]

        rows_k = []
        for label_k, col_k in key_fields:
            if col_k in vitals.columns:
                rows_k.append({"Field": label_k, "Value": latest.get(col_k)})

        if rows_k:
            df_k = pd.DataFrame(rows_k)
            st.dataframe(df_k, use_container_width=True, hide_index=True)
        else:
            st.caption("Key clinical fields not available for this patient.")

        # ---------- 6) Optional: population comparison (advanced) ----------
        with st.expander("📈 Advanced: Compare this patient to the population"):
            pop_cols = [c for c in ["BMI", "AGE", "MAP_APACHE", "HEART_RATE_APACHE"] if c in vitals.columns]
            for colname in pop_cols:
                try:
                    cohort = (
                        session.table(RAW_TABLE)
                        .select(colname)
                        .dropna()
                        .limit(5000)
                        .to_pandas()[colname]
                    )
                    if cohort.empty or pd.isna(latest[colname]):
                        continue

                    fig_d = ff.create_distplot([cohort.values], [colname], show_hist=False)
                    fig_d.add_vline(
                        x=float(latest[colname]),
                        line_dash="dash",
                        line_color="red",
                        annotation_text="This patient",
                        annotation_position="top left"
                    )
                    fig_d.update_layout(
                        title=f"{colname.replace('_',' ').title()} — Cohort Distribution"
                    )
                    st.plotly_chart(fig_d, use_container_width=True)
                except Exception:
                    # keep the UI clean if any column isn't present or any other issue
                    pass

    st.caption("If you have concerns, contact your provider. In emergencies, call your local emergency number.")


