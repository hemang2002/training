"""
STEP 3 - a web page for people who don't use curl.

The page does NOT contain the model. It sends the picture to the API (step 2) and shows the answer.

Run it (API from step 2 must be running in another terminal):
    streamlit run step3_simple_ui.py
"""
from pathlib import Path

import requests
import streamlit as st

API_URL = "http://127.0.0.1:5001/predict"       # where the API from step 2 is listening
SAMPLES = Path(__file__).resolve().parent.parent / "app" / "ui" / "samples"

st.title("🧠 What is in this picture?")
st.write("Pick a picture, press the button, and the model will guess.")

# ---------------------------------------------------------------- 1. choose a picture
uploaded = st.file_uploader("Upload your own picture", type=["png", "jpg", "jpeg"])
sample = st.selectbox("...or pick a sample", sorted(p.name for p in SAMPLES.glob("*.png")))

if uploaded is not None:
    image_bytes = uploaded.getvalue()
else:
    image_bytes = (SAMPLES / sample).read_bytes()

st.image(image_bytes, width=200)

# ---------------------------------------------------------------- 2. ask the API
if st.button("🔮 Ask the model", type="primary"):
    try:
        answer = requests.post(API_URL, files={"file": image_bytes}, timeout=10).json()
    except requests.exceptions.RequestException:  # API not running, or it sent back something that isn't JSON
        st.error("No answer from the API. Is it running?  Start it with:  python step2_predict_api.py")
        st.stop()

    if "label" not in answer:                    # the API answered, but with an error message
        st.error(f"The API said: {answer.get('error', answer)}")
        st.stop()

    # ------------------------------------------------------------ 3. show the answer
    st.success(f"I think it is a **{answer['label']}** ({answer['confidence'] * 100:.0f}% sure)")
    st.bar_chart(answer["all_probabilities"])
    st.caption("Each bar = how sure the model is about that class. All bars add up to 100%.")
