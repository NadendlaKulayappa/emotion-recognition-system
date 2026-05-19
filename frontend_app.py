import streamlit as st
import tempfile
from multimodal_ai_detection import analyze_interaction

st.set_page_config(page_title="Emotion AI", layout="centered")

st.title("🎯 Multimodal Emotion Detector")

mode = st.radio("Choose Input Mode", ["Live Capture", "Upload Files"])

text_input = st.text_area("Enter text (optional):")

audio_data = None
image_data = None

if mode == "Live Capture":
    audio_data = st.audio_input("🎤 Record voice")
    image_data = st.camera_input("📸 Capture face")
else:
    audio_data = st.file_uploader("Upload audio (.wav)", type=["wav"])
    image_data = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

# preview
if image_data:
    st.image(image_data, caption="Face Preview", width=300)

if audio_data:
    st.audio(audio_data)

if st.button("Analyze Emotion"):

    if not text_input and not audio_data:
        st.error("Provide text or audio")
    elif image_data is None:
        st.error("Provide image")
    else:
        try:
            audio_path = None

            if not text_input and audio_data:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    f.write(audio_data.getbuffer())
                    audio_path = f.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
                f.write(image_data.getbuffer())
                image_path = f.name

            result = analyze_interaction(
                audio_path=audio_path,
                image_path=image_path,
                text_input=text_input if text_input else None
            )

            st.subheader("📊 Results")

            c1, c2, c3 = st.columns(3)
            c1.metric("Text Emotion", result["text_emotion"])
            c2.metric("Face Emotion", result["face_emotion"])
            c3.metric("Final Emotion", result["final_emotion"])

            st.subheader("🗣 Transcript")
            st.write(result["text"])

            st.subheader("💬 AI Response")
            st.success(result["reply"])

        except Exception as e:
            st.error(str(e))