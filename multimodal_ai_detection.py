import os
import warnings
from datetime import datetime
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from groq import Groq
from deepface import DeepFace

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_NO_TF_IMPORT"] = "1"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= TEXT MODEL =================
tokenizer = AutoTokenizer.from_pretrained(
    "j-hartmann/emotion-english-distilroberta-base"
)

text_model = AutoModelForSequenceClassification.from_pretrained(
    "j-hartmann/emotion-english-distilroberta-base"
).to(device)


def get_text_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = text_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    return text_model.config.id2label[int(probs.argmax())]


# ================= AUDIO =================
def transcribe_groq(audio_path):
    with open(audio_path, "rb") as f:
        res = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=(audio_path, f, "audio/wav")
        )
    return res.text


# ================= FACE (IMPROVED) =================
def get_face_emotion_local(frame_path):
    try:
        emotions = []
        confidences = []

        for _ in range(5):  # 🔥 multi-frame voting
            result = DeepFace.analyze(
                img_path=frame_path,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )

            if isinstance(result, list):
                result = result[0]

            emotions.append(result['dominant_emotion'])
            confidences.append(max(result['emotion'].values()))

        # majority vote
        final = Counter(emotions).most_common(1)[0][0]
        avg_conf = sum(confidences) / len(confidences)

        # confidence filter
        if avg_conf < 40:
            return "uncertain"

        return final.lower()

    except:
        return "unknown"


# ================= FUSION =================
def fuse_emotions(text_emotion, face_emotion):

    if face_emotion in ["unknown", "uncertain"]:
        return text_emotion

    if text_emotion.lower() == face_emotion.lower():
        return text_emotion

    negative = ["sad", "angry", "fear", "disgust"]
    positive = ["joy", "happy"]

    if text_emotion.lower() in positive and face_emotion.lower() in negative:
        return face_emotion

    if text_emotion.lower() in negative and face_emotion.lower() in positive:
        return text_emotion

    if text_emotion.lower() == "neutral":
        return face_emotion

    return text_emotion


# ================= RESPONSE =================
def generate_response(text, final_emotion, text_emotion, face_emotion):

    # conflict-aware
    if text_emotion != face_emotion and face_emotion not in ["unknown", "uncertain"]:
        return "I sense a mismatch between what you're saying and how you're feeling. Do you want to talk about it?"

    responses = {
        "joy": "That’s great to hear! What made you feel so happy?",
        "sad": "I’m sorry you're feeling this way. Want to talk about it?",
        "angry": "Seems like something upset you. What happened?",
        "fear": "That sounds concerning. You're safe here.",
        "neutral": "You seem calm. Anything on your mind?"
    }

    return responses.get(final_emotion, "Tell me more about how you're feeling.")


# ================= LOG =================
def save_log(text, te, fe, final, reply):
    with open("session_log.txt", "a", encoding="utf-8") as f:
        f.write(
            f"\nTime: {datetime.now()}\n"
            f"User: {text}\n"
            f"Text Emotion: {te}\n"
            f"Face Emotion: {fe}\n"
            f"Final Emotion: {final}\n"
            f"AI: {reply}\n"
            f"{'-'*40}\n"
        )


# ================= PIPELINE =================
def analyze_interaction(audio_path=None, image_path=None, text_input=None):

    text = text_input if text_input else transcribe_groq(audio_path)

    text_emotion = get_text_emotion(text)
    face_emotion = get_face_emotion_local(image_path)
    final_emotion = fuse_emotions(text_emotion, face_emotion)

    reply = generate_response(text, final_emotion, text_emotion, face_emotion)

    save_log(text, text_emotion, face_emotion, final_emotion, reply)

    return {
        "text": text,
        "text_emotion": text_emotion,
        "face_emotion": face_emotion,
        "final_emotion": final_emotion,
        "reply": reply
    }