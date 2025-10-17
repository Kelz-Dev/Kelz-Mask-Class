import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
import plotly.graph_objects as go

# -------------------- Page Config --------------------
st.set_page_config(page_title="Face Mask Detection", page_icon="😷", layout="wide")

# -------------------- Custom CSS --------------------
st.markdown("""
    <style>
        .main {
            background-color: #0b0c10;
            color: #f8f8f8;
        }

        h1 {
            color: #00ffff;
            text-align: center;
            text-shadow: 0 0 20px #00ffff, 0 0 40px #0077ff;
        }

        .sub-text {
            text-align: center;
            color: #566573;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }

        /* Flex container for buttons */
        .button-row {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
            margin-top: 1rem;
        }

        /* Button styling */
        div.stButton > button {
            background-color: #1f2833;
            color: #00ffff;
            border-radius: 10px;
            border: 1px solid #00ffff;
            font-weight: 600;
            padding: 0.6rem 1.2rem;
            box-shadow: 0 0 10px #00ffff;
            transition: all 0.3s ease-in-out;
        }

        /* Predict button pulse animation */
        div.stButton > button[kind="primary"] {
            animation: pulseGlow 1.8s infinite alternate;
            border: 1px solid #00ffff;
        }

        @keyframes pulseGlow {
            from {
                box-shadow: 0 0 10px #00ffff, 0 0 20px #0077ff;
                transform: scale(1);
            }
            to {
                box-shadow: 0 0 25px #00ffff, 0 0 50px #0077ff;
                transform: scale(1.05);
            }
        }

        div.stButton > button:hover {
            background-color: #00ffff;
            color: #0b0c10;
            transform: scale(1.07);
            box-shadow: 0 0 30px #00ffff, 0 0 60px #0077ff;
        }

        .img-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 1.5rem;
        }

        .img-circle {
            width: 220px;
            height: 220px;
            border-radius: 50%;
            overflow: hidden;
            box-shadow: 0 0 25px #00ffff, 0 0 50px #0077ff;
            display: flex;
            justify-content: center;
            align-items: center;
            background: radial-gradient(circle at center, rgba(0,255,255,0.2), rgba(0,0,0,0.9));
            transition: box-shadow 0.3s ease-in-out, transform 0.3s ease-in-out;           
        }

        .img-circle img {
            width: 100%;
            height: 100%;
            border-radius: 50%;
            object-fit: cover;
        }

        .img-circle img:hover {
            transform: scale(1.05);
            box-shadow:0 0 40px #00ffff, 0 0 70px #0077ff; 
        }

        .round-image:hover {
            transform: scale(1.08);
            box-shadow: 0 0 45px #00ffff, 0 0 70px #0077ff;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #0b0c10;
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* Chart responsiveness */
        [data-testid="stPlotlyChart"] {
            width: 100% !important;
            height: auto !important;
        }

        /* Footer */
        footer {
            text-align: center;
            padding-top: 2rem;
            padding-bottom: 2rem;
            color: #888;
            font-size: 0.9rem;
        }

        @media (max-width: 768px) {
            h1 {
                font-size: 1.6rem;
            }
            div.stButton > button {
                width: 100%;
            }
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- Sidebar --------------------
st.sidebar.title("😷")
st.sidebar.markdown("## Model & Info", unsafe_allow_html=True)
st.sidebar.write("- **Input size**: 64 × 64 (HxW)")
st.sidebar.write("- **Classes**: Mask_correct, Mask_incorrect, No_Mask")
st.sidebar.markdown("---")
st.sidebar.write("👨‍💻 Developed by [Kelechi Alichi](https://github.com/Kelz-Dev/Data-Science-Projects)")

# -------------------- Main Title --------------------
st.markdown("<h1>😷 Face Mask Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Upload an image to detect mask usage with AI precision</p>", unsafe_allow_html=True)

# -------------------- Load Model --------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    model = tf.keras.models.load_model("mask_model.h5")
    return model

model = load_model()

# -------------------- Initialize Session --------------------
if 'chosen_image' not in st.session_state:
    st.session_state['chosen_image'] = None
if 'last_prediction' not in st.session_state:
    st.session_state['last_prediction'] = None
if 'uploaded_image' not in st.session_state:
    st.session_state['uploaded_image'] = None

# -------------------- Layout --------------------
col1, col2 = st.columns([2, 2])

with col1:
    st.markdown("<h2>Input</h2>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    # Reset session if image deleted
    if uploaded is None and st.session_state.get('uploaded_image') is not None:
        st.session_state['chosen_image'] = None
        st.session_state['last_prediction'] = None
        st.session_state['uploaded_image'] = None

    # Reset session if new image uploaded
    if uploaded  != st.session_state['uploaded_image']:
        st.session_state['chosen_image'] = None
        st.session_state['last_prediction'] = None
        st.session_state['uploaded_image'] = uploaded

    # Button row
    st.markdown("<div class='button-row'>", unsafe_allow_html=True)
    if uploaded is None:
        use_sample = st.button("Use Sample Image", type="primary")
    else:
        use_sample = False
    predict = st.button("Predict", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    # Image Handling
    chosen_image = None
    if use_sample:
        chosen_image = Image.open("img.jpg").convert("RGB")
        chosen_image = chosen_image.resize((256, 256), Image.Resampling.LANCZOS)
        st.session_state['chosen_image'] = chosen_image
        st.session_state['last_prediction'] = None

        # Display image (rounded and glowing)
        if st.session_state['chosen_image'] is not None:
            # st.markdown("<div class='img-container'><div class='img-circle'>", unsafe_allow_html=True)
            st.image(st.session_state['chosen_image'])
            # st.markdown("</div></div>", unsafe_allow_html=True)



    if uploaded is not None:
        chosen_image_upload = Image.open(uploaded).convert("RGB")
        st.session_state['chosen_image'] = chosen_image_upload

    # Prediction Logic
    if predict and st.session_state['chosen_image'] is not None:
        img = st.session_state['chosen_image']
        img_resized = img.resize((64, 64))
        arr = kimage.img_to_array(img_resized) / 255.0
        img_input = np.expand_dims(arr, axis=0)
        
        # st.markdown("<div class='img-container'><div class='img-circle'>", unsafe_allow_html=True)
        img_input_original = img.resize((256, 256), Image.Resampling.LANCZOS)
        # st.markdown("</div></div>", unsafe_allow_html=True)

        # Display image (rounded and glowing)
        if st.session_state['chosen_image'] is not None:
            # st.markdown("<div class='img-container'><div class='img-circle'>", unsafe_allow_html=True)
            st.image(img_input_original)
            # st.markdown("</div></div>", unsafe_allow_html=True)



        with st.spinner("Running prediction..."):
            preds = model.predict(img_input)
            probs = preds[0]
            classes = ["Mask_correct", "Mask_incorrect", "No_Mask"]
            top_idx = int(np.argmax(probs))
            label = classes[top_idx]
            confidence = float(probs[top_idx])

            st.session_state['last_prediction'] = {
                'label': label,
                'confidence': confidence,
                'probs': probs.tolist()
            }

    # Display Result Below Image
    if st.session_state.get('last_prediction') is not None:
        res = st.session_state['last_prediction']
        st.markdown(f"### 🧠 Result: **{res['label']}**")
        st.write(f"Confidence: **{res['confidence'] * 100:.2f}%**")

# -------------------- Display Chart --------------------
with col2:
    if st.session_state.get('last_prediction') is not None:
        res = st.session_state['last_prediction']
        fig = go.Figure(
            go.Bar(
                x=["Mask_correct", "Mask_incorrect", "No_Mask"],
                y=res['probs'],
                marker_color=["#16A085", "#F59E0B", "#EF4444"]
            )
        )
        fig.update_layout(
            title_text="Prediction Probabilities",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color="#E6EEF8"
        )
        st.plotly_chart(fig, use_container_width=True)

# -------------------- Footer --------------------
st.write("<footer>This model makes mistakes, but it has a 90% accuracy</footer>", unsafe_allow_html=True)
