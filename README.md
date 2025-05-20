# ResuAI
Voice interview agent and CV shortlisting system using LLMs (LLaMA via Ollama), Whisper, and TTS

# 🧠 ResuAI – AI-Powered CV Screening & Voice Interview Agent

ResuAI is an AI-driven toolkit to automate the recruitment process. It consists of two powerful Streamlit apps:

1. **Enhanced Resume Screening** (`enhanced_resu_ai.py`) – Upload CVs and a job description to automatically score and shortlist candidates using a local LLM (LLaMA via Ollama).
2. **Voice Interview Agent** (`voice_agent.py`) – Conduct voice-based interviews with shortlisted candidates, validate their answers using AI, and generate an automated evaluation.

---

## 🚀 Features

### ✅ Enhanced Resume Screening (`enhanced_resu_ai.py`)
- Supports `.pdf` and `.docx` resumes.
- Extracts contact info (Name, Email, Phone).
- Evaluates CVs against a given job description using LLaMA via Ollama.
- Shortlists candidates based on a scoring threshold.
- Interactive results table and downloadable CSV.

### 🎤 Voice Interview Agent (`resuai_with_voice_agent.py`)
- Conducts structured interviews via text-to-speech and microphone.
- Records and transcribes candidate answers using Whisper.
- Validates answers and scores interviews using LLaMA via Ollama.
- Saves full interview results to CSV.

### 🧩 All-in-One Agent (`ResuAI_combined.py`)
- Combines resume screening and voice interviews in a single, seamless Streamlit app.
- Allows you to screen CVs, extract top candidates, and immediately conduct interviews — all from one interface.
- No need to manually link files or move data between steps.

---

### ⚙️ Installation

# 1. Clone the repository

_git clone https://github.com/yourusername/resuai.git
cd resuai_

# 2. Install required dependencies:**

Make sure you have Python 3.9+ installed.
_pip install -r requirements.txt_

_Note: You also need to install ffmpeg, portaudio, and Ollama._

# 3. Install & run Ollama with LLaMA3:**
**Install Ollama if you haven't:**
_https://ollama.com/download_

**In your terminal:**
_ollama run llama3_

# ▶️ Running the Apps
**1. All-in-One Unified App**

_streamlit run ResuAI_combined.py_

**2. Resume Screener**

_streamlit run enhanced_resu_ai.py_

**3. Voice Interview Agent**
Make sure to modify the CSV path in voice_agent.py (line 13 & 14) to point to your candidates.csv.

_streamlit run voice_agent.py_

# 📁 Folder Structure

resuai/
├── enhanced_resu_ai.py          # Resume screening app
├── resuai_with_voice_agent.py   # Voice interview agent
├── ResuAI_combined.py           # Unified resume + interview app
├── requirements.txt             # Dependencies
└── README.md


# 📌 Notes
This project assumes Ollama is running locally at _http://localhost:11434._

Audio transcription is done using faster-whisper, and speech is generated with the TTS library.

Microphone and speaker access is required for the voice agent.

Candidate information should be stored in a CSV file like:

_Name,Email,Phone
Alice Johnson,alice@example.com,1234567890
Bob Smith,bob@example.com,0987654321_

# 🛡 License
This project is open-source and licensed under the MIT License.

# 🤝 Contributions
Pull requests are welcome! If you have suggestions or improvements, feel free to fork and submit a PR.
