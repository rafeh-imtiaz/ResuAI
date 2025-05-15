import streamlit as st
import pandas as pd
import numpy as np
import sounddevice as sd
from datetime import datetime
from faster_whisper import WhisperModel
from TTS.api import TTS
import ollama
import os
import re

# ✅ Set Streamlit Page Config
st.set_page_config(page_title="Voice Interview Agent", layout="centered")

# -------------------------------
# 🔧 Config
# -------------------------------
CANDIDATES_CSV_PATH = r"C:\Users\JBS\Desktop\candidates.csv"
SAVE_FOLDER_PATH = r"C:\Users\JBS\Desktop"
RESULTS_CSV = os.path.join(SAVE_FOLDER_PATH, "interview_results.csv")

# -------------------------------
# 🧠 Load Models
# -------------------------------
@st.cache_resource
def load_models():
    whisper_model = WhisperModel("tiny", compute_type="int8")
    tts_model = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=False, gpu=False)
    return whisper_model, tts_model

whisper_model, tts_model = load_models()

# -------------------------------
# 🔉 Audio Playback
# -------------------------------
def play_audio(text):
    audio = tts_model.tts(text=text)
    sd.play(np.array(audio, dtype=np.float32), samplerate=22050)
    sd.wait()

# -------------------------------
# 🤖 LLaMA Validator & Scoring
# -------------------------------
def validate_answer(question, answer):
    prompt = f"Is the following answer relevant to the question?\nQuestion: {question}\nAnswer: {answer}\nRespond with 'Yes' or 'No' only."
    result = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    return result["message"]["content"].strip()

def score_interview(data):
    qas = "\n".join([f"Q: {q}\nA: {a}" for q, a in data.items() if q.startswith("What") or q.startswith("Are")])
    prompt = f"""
    You are an HR expert. Based on the following answers, give a relevance score from 0 to 100 and provide a short, clear explanation of why you gave that score:
    {qas}
    Score and explanation:
    """
    result = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    
    score_text = result["message"]["content"].strip()
    
    match = re.search(r"(\d{1,3})", score_text)
    score = int(match.group()) if match else "Invalid score"
    explanation = score_text[match.end():].strip() if match else "No explanation provided"
    
    return score, explanation

def analyze_sentiment(all_answers):
    joined_answers = " ".join(all_answers)
    prompt = f"""
    Analyze the sentiment of the following text (a job interview transcript). Respond with a single word: Positive, Neutral, or Negative — and give a brief explanation.
    
    Transcript: {joined_answers}
    
    Sentiment and Explanation:
    """
    result = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    
    sentiment_text = result["message"]["content"].strip()
    
    match = re.search(r"(Positive|Neutral|Negative)", sentiment_text, re.IGNORECASE)
    sentiment = match.group() if match else "Unknown"
    explanation = sentiment_text[match.end():].strip() if match else "No explanation found."
    
    return sentiment.capitalize(), explanation

# -------------------------------
# 🎤 Audio Recording
# -------------------------------
def record_audio(duration=10, fs=16000):
    st.info("🎙 Recording... Please answer now.")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    st.success("✅ Recording complete.")
    return audio[:, 0]

def transcribe_audio(audio):
    segments, _ = whisper_model.transcribe(audio, language="en")
    return " ".join([seg.text for seg in segments])

# -------------------------------
# 🚀 Main App
# -------------------------------
def main():
    st.title("🗣 ResuAI Voice Interviewer")

    # Final check to save at end
    if "interview_data" in st.session_state and "current_index" in st.session_state:
        if st.session_state.current_index >= len(st.session_state.candidates):
            st.success("✅ All interviews complete!")
            df = pd.DataFrame(st.session_state.interview_data)
            try:
                df.to_csv(RESULTS_CSV, index=False)
                st.write("Results saved to:")
                st.code(RESULTS_CSV)

                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Interview Results as CSV",
                    data=csv_bytes,
                    file_name="interview_results.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"❌ Failed to save results: {e}")
            return

    # Session state init
    if "candidates" not in st.session_state:
        st.session_state.candidates = pd.read_csv(CANDIDATES_CSV_PATH).to_dict(orient="records")
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "interview_data" not in st.session_state:
        st.session_state.interview_data = []

    # Current candidate
    candidate = st.session_state.candidates[st.session_state.current_index]
    st.subheader(f"Interviewing: {candidate['Name']}")

    questions = [
        "What position are you applying for?",
        "Can you briefly describe your previous work experience?",
        "What is your highest qualification?",
        "Are you available for remote work?",
        "What are your salary expectations?"
    ]

    if "step" not in st.session_state:
        st.session_state.step = 0
        st.session_state.answers = {}
        st.session_state.greeted = False

    # Play greeting at beginning of each candidate
    if st.session_state.step == 0 and not st.session_state.get("greeted", False):
        greeting = f"Hello {candidate['Name']}, this is HR from Blutech Consulting. Let’s begin. {questions[0]}"
        play_audio(greeting)
        st.session_state.greeted = True

    if st.session_state.step < len(questions):
        current_q = questions[st.session_state.step]
        st.markdown(f"### ❓ {current_q}")
        if st.button("🎙 Record Response"):
            audio = record_audio()
            text = transcribe_audio(audio)
            st.write(f"🗣 You said: **{text}**")

            valid = validate_answer(current_q, text)
            if "no" in valid.lower():
                st.warning("⚠️ That didn’t seem relevant. Please try again.")
                play_audio(f"That didn’t seem relevant. {current_q}")
            else:
                st.session_state.answers[current_q] = text
                st.session_state.step += 1

                if st.session_state.step < len(questions):
                    play_audio(questions[st.session_state.step])
                else:
                    # Interview complete — collect metadata and score
                    st.session_state.answers["Candidate Name"] = candidate["Name"]
                    st.session_state.answers["Email"] = candidate["Email"]
                    st.session_state.answers["Phone"] = candidate["Phone"]

                    score, explanation = score_interview(st.session_state.answers)
                    st.session_state.answers["Score"] = score
                    st.session_state.answers["Explanation"] = explanation

                    all_answer_texts = list(st.session_state.answers.values())
                    sentiment, sentiment_expl = analyze_sentiment(all_answer_texts)
                    st.session_state.answers["Sentiment"] = sentiment
                    st.session_state.answers["Sentiment Explanation"] = sentiment_expl

                    st.session_state.interview_data.append(st.session_state.answers)

                    # Prepare for next candidate
                    st.session_state.step = 0
                    st.session_state.current_index += 1
                    st.session_state.greeted = False

                    if st.session_state.current_index < len(st.session_state.candidates):
                        st.rerun()
                    else:
                        df = pd.DataFrame(st.session_state.interview_data)
                        try:
                            df.to_csv(RESULTS_CSV, index=False)
                            st.success("✅ All interviews complete!")
                            st.write("Results saved to:")
                            st.code(RESULTS_CSV)

                            csv_bytes = df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="📥 Download Interview Results as CSV",
                                data=csv_bytes,
                                file_name="interview_results.csv",
                                mime="text/csv"
                            )
                        except Exception as e:
                            st.error(f"❌ Failed to save results: {e}")
                    return

if __name__ == "__main__":
    main()
