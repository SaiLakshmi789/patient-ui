from snowflake.snowpark import Session
import streamlit as st  # if this is a Streamlit-based project

SF = st.secrets["snowflake"]

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

session = get_session()
