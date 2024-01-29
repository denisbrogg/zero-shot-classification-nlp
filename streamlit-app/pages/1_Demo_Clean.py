import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import sys

sys.path.append("..")
from zsl.components import zero_shot_predict


labels_as_text = st.text_area(
    label="Which categories do you want to predict?",
    placeholder="Insert here your prediction labels separated by comma",
)

if len(labels_as_text):
    identified_labels = [s.strip() for s in labels_as_text.split(",")]
    labels = st.multiselect(
        label="Identified Labels", options=identified_labels, default=identified_labels
    )

selected_model = st.selectbox(
    label="Embedder", options=["sentence-transformers/all-MiniLM-L6-v2"]
)

with st.spinner("Loading embedder..."):
    model = SentenceTransformer(selected_model)

query = st.text_input(label="Input", placeholder="Insert here the input to classify")

if len(query) and len(labels):
    prediction = zero_shot_predict(query, labels=labels, model=model)[0]
    st.write(f"#### Predicted label: {prediction['label']}")

    chart_data = pd.DataFrame(
        np.eye(len(labels)) * prediction["similarities"][0],
        columns=labels,
        index=labels,
    )

    st.bar_chart(
        chart_data,
    )
