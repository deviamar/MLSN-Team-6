import streamlit as st
import subprocess
import sys

st.title("Real-Time Emotion Detection")
st.write("Press start to run the model!")

if st.button("Start Model"):
    st.write("Model Starting")

    subprocess.Popen([sys.executable, "runModel.py"])

    st.success("Model is opening")