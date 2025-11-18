import os, time, urllib.parse
import jwt  # PyJWT
from Snowflake import session
import pandas as pd
# import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,recall_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import joblib
import json
from pathlib import Path

# Where to store model artifacts
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)  # create if not exists

MODEL_PATH = MODELS_DIR / "rf_model.joblib"
COLS_PATH  = MODELS_DIR / "feature_columns.json"
SCALER_PATH = MODELS_DIR / "scaler.joblib"   # optional, but good to have


import sys, os
try:
    # Python 3.7+ 
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    # Fallback for older environments
    os.environ["PYTHONIOENCODING"] = "utf-8"


#Add these helper constants near the top (edit the URL)

# Point this to your deployed Streamlit app (or http://localhost:8501 while testing)
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8501")

# Keep this secret out of source control; override via environment or secrets manager
JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_ME_TO_A_LONG_RANDOM_SECRET")

table="Employee"

# Assuming you have a session object ready
def get_raw_data(session):
    raw_data = session.sql("""
        SELECT encounter_id, patient_id, hospital_id, age, bmi, weight, apache_2_diagnosis,
               apache_3j_diagnosis, apache_post_operative, gender, height, icu_admit_source, 
               icu_stay_type, intubated_apache, heart_rate_apache, map_apache, 
               resprate_apache, temp_apache, ventilated_apache, hospital_death
        FROM ETL_DB.ETL_SCH.RAW
        WHERE hospital_death IS NOT NULL 
    """).to_pandas()
    raw_data.head()

    return raw_data

def get_ldg_data(session):
    ldg_data = session.sql("""
        SELECT encounter_id, patient_id, hospital_id, age, bmi, weight, apache_2_diagnosis,
               apache_3j_diagnosis, apache_post_operative, gender, height, icu_admit_source, 
               icu_stay_type, intubated_apache, heart_rate_apache, map_apache, 
               resprate_apache, temp_apache, ventilated_apache, hospital_death
        FROM ETL_DB.ETL_SCH.LDG
    """).to_pandas()

    print(3)

    ldg_data.head()

    return ldg_data

# Assuming the data is already extracted into a pandas dataframe
def train_rf_model(df):
 
    print(2)
    import pandas as pd
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
 
    print(2)
    # ----------------------------
    # 1️⃣ Display & Cleaning Setup
    # ----------------------------
    pd.set_option('display.max_rows', 85)
    drop_cols = [
        'Unnamed: 83', 'ethnicity', 'icu_admit_source', 'icu_id',
        'icu_stay_type', 'icu_type', 'encounter_id', 'patient_id', 'hospital_id'
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    df
    print(2.1)
 
    # ----------------------------
    # 2️⃣ Basic Null Value Overview
    # ----------------------------
    null_pct = (df.isnull().sum() / len(df)) * 100
    total_nulls = df.isnull().sum().sum()
    print("Null value percentage by column:\n", null_pct[null_pct > 0].sort_values(ascending=False).head(10))
    print(f"\nTotal missing values in dataset: {total_nulls}")
    print(2.2)
 
    # ----------------------------
    # 3️⃣ Handle Missing Values
    # ----------------------------
    # Fill numeric columns with median, categorical with mode
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
 
    # Confirm no NaNs remain
    print(f"✅ Remaining NaNs after fill: {df.isnull().sum().sum()}")
 
    # ----------------------------
    # 4️⃣ One-Hot Encoding
    # ----------------------------
    ds = pd.get_dummies(data=df, drop_first=True)
    ds = ds.replace({True: 1, False: 0})
    ds.infer_objects(copy=False)
    print(2.3)
 
    # ----------------------------
    # 5️⃣ Feature/Label Split
    # ----------------------------
    X = ds.drop('HOSPITAL_DEATH', axis=1)
    y = ds['HOSPITAL_DEATH']
    print(2.4)
 
    # ----------------------------
    # 6️⃣ Apply SMOTE Oversampling
    # ----------------------------
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(2.5)
 
    # ----------------------------
    # 7️⃣ Split Train / Validation / Test
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, stratify=y_res, test_size=0.25, shuffle=True, random_state=42
    )
    X_val, X_tst, y_val, y_tst = train_test_split(
        X_test, y_test, stratify=y_test, test_size=0.5, shuffle=True, random_state=42
    )
 
    print("\nDataset shapes:")
    print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_tst.shape}")
    print("\nLabel counts (train/val):")
    print(pd.Series(y_train).value_counts(), "\n", pd.Series(y_val).value_counts())
 
    # ----------------------------
    # 8️⃣ MinMax Scaling
    # ----------------------------
    scaler = MinMaxScaler()
    Xt_scaled = scaler.fit_transform(X_train)
    Xv_scaled = scaler.transform(X_val)
    Xts_scaled = scaler.transform(X_tst)
 
    # ----------------------------
    # 9️⃣ Train Random Forest
    # ----------------------------
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(Xt_scaled, y_train)
 
    # ----------------------------
    # 🔟 Evaluate Model
    # ----------------------------
    y_pred = rf_model.predict(Xts_scaled)
    accuracy = accuracy_score(y_tst, y_pred)
    print(f"\n✅ Model accuracy: {accuracy * 100:.2f}%")
    recall = recall_score(y_tst, y_pred)
    print(f"\n✅ Model recall: {recall * 100:.2f}%")

    
    # ----------------------------
    # 1️⃣1️⃣  SAVE ARTIFACTS
    # ----------------------------
    feature_cols = list(X.columns)  # columns BEFORE scaling

    # Save RF model
    joblib.dump(rf_model, MODEL_PATH)
    print(f"💾 Saved RandomForest model to: {MODEL_PATH}")

    # Save scaler (optional but useful if you reuse it later)
    joblib.dump(scaler, SCALER_PATH)
    print(f"💾 Saved scaler to: {SCALER_PATH}")

    # Save feature column names for SHAP / Streamlit app
    with open(COLS_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"💾 Saved feature column list to: {COLS_PATH}")


    return rf_model, X_train, y_train, X_val, y_val, X_tst, y_tst

def predict_and_add_predictions(session, rf_model, data,features, target):
    

    def send_email(patient_id):
        print(patient_id)
        # 1) get email exactly like before (collect 1 row)
        df_email = session.sql(
            "select email from ETL_DB.ETL_SCH.PATIENTS where patient_id = ?",
            params=[str(patient_id)]
        ).collect()
        if not df_email or not df_email[0][0]:
            print(f"No email found for patient_id={patient_id}")
            return
        to_email = str(df_email[0][0])

        # 2) mint JWT (force str in case your PyJWT returns bytes)
        payload = {"sub": str(patient_id), "scope": "patient_dashboard", "exp": int(time.time()) + 3600}
        token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
        if isinstance(token, bytes):
            token = token.decode("utf-8")

        # 3) build link
        link = f"{APP_BASE_URL}?token={urllib.parse.quote(token)}"

        # 4) subject/body (plain strings)
        subject = "⚠ Health Risk Alert: Please review your results"
        body = (
            "Based on your recent reports, you may be at higher health risk. "
            "Please review your secure dashboard and contact your Primary Care Provider.\n\n"
            f"Open your dashboard: {link}\n\n"
            "If you need assistance finding a provider or scheduling an appointment, we’re here to help: 1-800-000-0000\n"
            "Take care,\nProActiveCare"
        )

        # 5) call exactly like your old style (string literals)
        send_email_df = session.sql(
            f"CALL SYSTEM$SEND_EMAIL('alert_email_int','{to_email}','{subject}','{body}')"
        ).collect()
        print("Email sent:", to_email)
        
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Extract the features for prediction
    features_for_prediction = data.drop(columns=['HOSPITAL_DEATH', 'ENCOUNTER_ID', 'PATIENT_ID', 'HOSPITAL_ID'])
    print(7)
    # Make predictions on the raw data
    # Step 1: Encode training data
    import pandas as pd
    X = pd.get_dummies(features, drop_first=True)
    y = target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model.fit(X_train, y_train)

    # Step 2: Encode prediction data the same way
    features_for_prediction = pd.get_dummies(features_for_prediction, drop_first=True)

    # Step 3: Match the column structure to training data
    features_for_prediction = features_for_prediction.reindex(columns=X_train.columns, fill_value=0)

    # Step 4: Predict safely
    predictions = rf_model.predict(features_for_prediction)
    print(8)
    # Add the predictions to the original dataframe
    data['HOSPITAL_DEATH'] = predictions
    print(predictions)
    print(9)

    from snowflake.snowpark.functions import lit, current_timestamp

    core_columns = ['ENCOUNTER_ID', 'PATIENT_ID', 'HOSPITAL_ID','HOSPITAL_DEATH']
    df_to_upload = data[core_columns].copy() 

    # 2. Convert to Snowpark DataFrame (This defines 'source_df')
    source_df = session.create_dataframe(df_to_upload) 

    # 3. Define the Target Table object (This defines 'target_table')
    snowflake_table_name = "ETL_DB.ETL_SCH.STD" 
    target_table = session.table(snowflake_table_name)

    # 1. Define the Join Expression
    join_condition = (target_table['ENCOUNTER_ID'] == source_df[ 'ENCOUNTER_ID'])
    from snowflake.snowpark.functions import when_matched, when_not_matched   # Action 1: WHEN MATCHED (Update existing record to inactive 'N')
    # Line 121:
    matched_action = when_matched() \
        .update({"ACTIVE_FLAG": lit('N'), "UPDT_TIMESTMP": current_timestamp()})

    not_matched_action = when_not_matched() \
        .insert({
            "ENCOUNTER_ID": source_df['ENCOUNTER_ID'],
            "PATIENT_ID": source_df['PATIENT_ID'],
            "HOSPITAL_ID": source_df['HOSPITAL_ID'],
            "HOSPITAL_DEATH": source_df['HOSPITAL_DEATH'],
            "INSERT_TIMESTMP": current_timestamp(),
            "ACTIVE_FLAG": lit('Y') 
        })

    # Define ALL MERGE ACTIONS in a single list
    merge_actions = [matched_action, not_matched_action]

    # Execute the MERGE
    merge_result = target_table.merge(
        source_df,          
        join_condition,     
        merge_actions       
    )

    print(f"\n✅ MERGE operation complete on {snowflake_table_name}.")
    print(f"   Rows updated (set to 'N'): {merge_result.rows_updated}")
    print(f"   Rows inserted (set to 'Y'): {merge_result.rows_inserted}")
    print(10)

    print(10.5)
    source_df=source_df.to_pandas()
    print(source_df)
    # assuming you have an ID column to track patients
    for i, pred in enumerate(predictions):
        if pred == 1:
            patient_id = source_df.iloc[i]['PATIENT_ID']
            send_email(patient_id)

def train_and_predict_hospital_death(session):
        # Step 1: Get raw data
        raw_data = get_raw_data(session)
        
        # Step 2: Train the model
        rf_model, X_train, y_train, X_val, y_val, X_tst, y_tst = train_rf_model(raw_data)


        ldg_data = get_ldg_data(session)
    
        # Step 3: Predict and add predictions back to raw table
        predict_and_add_predictions(session, rf_model, ldg_data,X_train, y_train)

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def s3tolndng(session,table):

    df = session.sql("LIST @PAC_STAGE")
    df.show()
    df = session.sql("""
        INSERT INTO ETL_DB.ETL_SCH.FILE_HISTORY_TEMP (file_name, file_size, md5, last_modified)
        SELECT 
            "name", 
            "size", 
            "md5", 
            TO_TIMESTAMP("last_modified", 'DY, DD MON YYYY HH24:MI:SS GMT')
        FROM TABLE(RESULT_SCAN(LAST_QUERY_ID())) AS t
    """)    

    df.show()
    df_MD5 = session.sql("SELECT md5 FROM ETL_DB.ETL_SCH.FILE_HISTORY_TEMP where md5 not in (SELECT md5 FROM ETL_DB.ETL_SCH.FILE_HISTORY)")

    new_files = session.sql("""
        SELECT file_name
        FROM ETL_DB.ETL_SCH.FILE_HISTORY_TEMP where md5 not in (SELECT md5 FROM ETL_DB.ETL_SCH.FILE_HISTORY);
    """).collect()

    if not new_files:  # Check if new_files is empty (None or [])
        print("No new files found to process.")
          # Skip the rest of the operations if no new files

    else:
        print(f"Found {len(new_files)} new files to process.")
        for row in new_files:
            file = row['FILE_NAME'].replace('s3://etlbucket00017/ProActiveCare/', 'PAC_STAGE/')
            session.sql(f"COPY INTO ETL_DB.ETL_SCH.LDG FROM @{file} FILE_FORMAT=(FORMAT_NAME='csv_format')").collect()
        print("1")
        # session.sql("INSERT INTO ETL_DB.ETL_SCH.RAW SELECT *, CURRENT_TIMESTAMP FROM ETL_DB.ETL_SCH.LDG").collect()
        # print("2")
        # Execute the function
        train_and_predict_hospital_death(session)

    df = session.sql("INSERT INTO ETL_DB.ETL_SCH.FILE_HISTORY (file_name, file_size,MD5, last_modified) SELECT * FROM ETL_DB.ETL_SCH.FILE_HISTORY_TEMP where md5 not in (SELECT md5 FROM ETL_DB.ETL_SCH.FILE_HISTORY)")
    df.show()

    LDG_TRUNCATE=session.sql("truncate ETL_DB.ETL_SCH.LDG ")
    LDG_TRUNCATE.show()

    FH_TRUNCATE=session.sql("truncate ETL_DB.ETL_SCH.FILE_HISTORY_TEMP ")
    FH_TRUNCATE.show()

    

s3tolndng(session,table)