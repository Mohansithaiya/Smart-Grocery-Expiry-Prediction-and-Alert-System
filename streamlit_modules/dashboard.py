# Dashboard visualization

import streamlit as st
import pandas as pd


def display_dashboard(df):
    st.subheader("Expiry Dashboard")
    st.dataframe(df)
