import streamlit as st
import requests
from PIL import Image
import io

# --- CONFIG ---
API_URL = "http://localhost:8000/predict"

# SET LAYOUT TO WIDE (Uses the whole screen)
st.set_page_config(page_title="Tea Leaf Lens", page_icon="🍃", layout="wide")

# --- UI HEADER ---
st.title("🍃 Tea Leaf Lens")
st.markdown("### AI-Powered Disease Detection System")

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # CREATE TWO COLUMNS (Left for Image, Right for Results)
    col1, col2 = st.columns([1, 1])  # Equal width columns

    with col1:
        st.subheader("Your Image")
        image = Image.open(uploaded_file)
        # Display image with a reasonable width, but centered in the column
        st.image(image, caption='Uploaded Leaf', width=450)

    with col2:
        st.subheader("Analysis Results")

        # Add a "Predict" button
        if st.button("🔍 Analyze Leaf", type="primary"):
            with st.spinner("Consulting the AI Model..."):
                try:
                    # Prepare the file
                    uploaded_file.seek(0)
                    files = {"file": uploaded_file}

                    # Send Request
                    response = requests.post(API_URL, files=files)

                    if response.status_code == 200:
                        data = response.json()
                        pred = data["prediction"]
                        conf = data["confidence"]

                        # --- DISPLAY BIG METRICS ---
                        if pred == "Healthy":
                            st.success(f"### 🌱 Result: {pred}")
                        else:
                            st.error(f"### ⚠️ Disease: {pred}")

                        st.info(f"**Confidence Score:** {conf}")

                        # --- DETAILED SCORES ---
                        st.markdown("#### Detailed Probability Map:")
                        st.json(data["probabilities"])

                    else:
                        st.error("Error: Could not connect to the Backend API.")

                except Exception as e:
                    st.error(f"Connection Error: {e}")