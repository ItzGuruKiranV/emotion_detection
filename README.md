
# 🧠 AI-Powered Multimodal Emotion Analysis Platform

A Python-based Streamlit application that analyzes emotion-related signals from facial images, audio, and text. It processes each modality independently and combines available predictions through a multimodal late-fusion system.

> **Note:** This is an educational project. Predictions are estimates and do not establish a person's actual internal emotional state. Do not use this application for medical diagnosis or other high-stakes decisions.

## ✨ Features

- **Facial Emotion Detection:** Uses OpenCV Haar Cascade for face detection and a MobileNetV2-based classifier.
- **Audio Emotion Analysis:** Extracts acoustic features such as MFCCs, pitch-related features, RMS energy, and zero-crossing rate using Librosa.
- **Text Emotion Detection:** Uses a fine-tuned BERT sequence-classification model.
- **Multimodal Fusion:** Combines available modality predictions and can identify disagreements between them.
- **Uncertainty Estimation:** Displays prediction scores and estimates uncertainty from the output probabilities.
- **Prediction History:** Uses SQLite to store prediction results and related metadata.
- **Interactive Dashboard:** Provides a Streamlit interface for image, audio, text, and combined emotion analysis.
- **Automated Testing:** Uses Pytest to test core components.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| Computer Vision | OpenCV, TensorFlow/Keras, MobileNetV2 |
| Audio Processing | Librosa, NumPy, SoundFile |
| Text Classification | PyTorch, Hugging Face Transformers, BERT |
| Multimodal Fusion | Python, probability-based late fusion |
| Database | SQLite |
| Testing | Pytest |

## 🏗️ System Architecture

The application follows a modular architecture:

1. **Input:** The user provides an image, audio clip, text, or a combination of inputs.
2. **Preprocessing:** Each input is validated and transformed into the format expected by its model or classifier.
3. **Modality Analysis:**
   - Vision analyzes facial images.
   - Audio extracts acoustic features and applies a rule-based classifier.
   - Text uses BERT to classify the input text.
4. **Emotion Label Mapping:** Predictions are mapped to the application's shared emotion categories.
5. **Late Fusion:** Available predictions are combined into a final result.
6. **Result Storage:** Prediction information can be stored in SQLite and displayed in the dashboard.

### Emotion Categories

The application's canonical emotion labels are:

`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`

Not every modality directly predicts all seven categories in the same way. Text-model labels are mapped into the shared label set.

## 📁 Project Structure

```text
emotion_detection/
├── app.py
├── requirements.txt
├── README.md
├── PROJECT_GUIDE.md
├── classes.json
├── models/
│   ├── best_emotion_model.h5
│   └── haarcascade_frontalface_default.xml
├── results/
│   └── checkpoint-1600/
├── src/
│   ├── config.py
│   ├── vision/
│   ├── audio/
│   ├── text/
│   ├── fusion/
│   ├── database/
│   ├── evaluation/
│   └── utils/
├── scripts/
│   ├── setup.ps1
│   ├── run.ps1
│   ├── test.ps1
│   └── evaluate.ps1
├── tests/
├── data/
│   └── sample/
└── docs/
```

Some folders or scripts may differ depending on the version of the repository.

## 💻 Installation and Setup — Windows

### Prerequisites

- Windows 10 or Windows 11
- Python 3.10 or 3.11 recommended
- Git, if cloning the repository
- Sufficient RAM and disk space for the machine-learning dependencies and model files

### 1. Open PowerShell

Navigate to the project directory:

```powershell
cd C:\Gurukiran\projects\emotion_detection
```

If your project is stored elsewhere, replace the path with your actual project location.

### 2. Set Up the Python Environment

If the project contains the setup script, you can use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup.ps1
```

Alternatively, create a virtual environment manually:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you already have a working `venv` or `.venv` environment, reuse it instead of creating another one.

> First-time installation may take several minutes because the project uses machine-learning libraries.

## 🚀 Running the Application

Run the following commands from the project root directory.

### Option 1: Use the Run Script

If `scripts\run.ps1` exists:

```powershell
.\scripts\run.ps1
```

### Option 2: Start Streamlit Directly

For an environment named `venv`:

```powershell
.\venv\Scripts\streamlit.exe run app.py --server.port 8501
```

For an environment named `.venv`:

```powershell
.\.venv\Scripts\streamlit.exe run app.py --server.port 8501
```

Once Streamlit starts, open:

**http://localhost:8501**

Keep the terminal open while using the application.

To stop the application, press:

`Ctrl + C`

### If Port 8501 Is Already in Use

First, check whether the application is already running at `http://localhost:8501`.

If you need to start another instance, use port `8502`:

```powershell
.\venv\Scripts\streamlit.exe run app.py --server.port 8502
```

Then open:

**http://localhost:8502**

If your environment is named `.venv`, replace `venv` in the command with `.venv`.

## 🧪 Running Tests

If the project contains the test script, run:

```powershell
.\scripts\test.ps1
```

Alternatively, run Pytest directly:

```powershell
.\venv\Scripts\pytest.exe -v
```

For a `.venv` environment:

```powershell
.\.venv\Scripts\pytest.exe -v
```

Check the terminal output to confirm whether the tests pass.

## 📊 Running Model Evaluation

If the evaluation scripts are available, run:

```powershell
.\scripts\evaluate.ps1
```

Alternatively:

```powershell
.\venv\Scripts\python.exe scripts\evaluate.py
```

Review the generated evaluation report to understand the model's performance.

Accuracy and F1 scores depend on the evaluation dataset and methodology. Do not interpret validation results as independent test-set performance unless the evaluation confirms that distinction.

## 🔍 Understanding the Models

### 1. Vision — MobileNetV2

The vision pipeline detects a face using OpenCV Haar Cascade, preprocesses the image, and passes it to the MobileNetV2-based classifier.

### 2. Audio — Acoustic Feature Analysis

The audio pipeline uses Librosa to extract features such as MFCCs, pitch-related statistics, RMS energy, and zero-crossing rate.

The current audio classifier uses heuristic rules to estimate emotion probabilities. It is not an end-to-end trained deep-learning speech emotion model.

### 3. Text — BERT

The text pipeline tokenizes the input and passes it to a fine-tuned BERT sequence classifier. Its output labels are mapped to the application's shared emotion categories.

### 4. Multimodal Late Fusion

Each modality generates its own prediction. The fusion module combines available predictions using probability-based weighting and can flag disagreement or uncertainty.

Late fusion allows the system to work with different combinations of inputs instead of requiring all three modalities for every prediction.

## 🗄️ Database and Privacy

The application uses SQLite to maintain prediction history and related metadata.

The project is designed to store prediction information rather than raw media in the history table. However, review the current implementation and configuration before processing sensitive inputs.

Use the history-clearing feature where available, and avoid uploading private images, recordings, or text without consent.

## 🛠️ Troubleshooting

### Port 8501 Is Already in Use

Check whether the app is already running. Otherwise, launch it on port `8502`.

### PowerShell Blocks Script Execution

Run this in the current PowerShell session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then retry the script.

### Model or Checkpoint Not Found

- Run commands from the project root.
- Confirm the required model files exist.
- Check that dependencies are installed in the correct virtual environment.
- Read the full terminal error before changing packages or deleting files.

### Application Is Slow

Loading TensorFlow and BERT models can require significant memory and time. Close unnecessary applications and allow model loading to finish.

### Predictions Are Inconsistent

Predictions may be affected by image quality, lighting, background noise, short audio clips, ambiguous text, model limitations, and the audio classifier's heuristic rules.

Test each modality independently before interpreting a combined prediction.

## ⚠️ Limitations

- Emotion predictions are estimates and may be incorrect.
- Facial expressions and vocal characteristics do not prove a person's internal emotional state.
- The audio classifier uses handcrafted features and heuristic rules.
- Mapping different model label sets into one taxonomy can lose information.
- Performance depends on input quality, model weights, hardware, and evaluation methodology.

This project is intended for educational and demonstration purposes, not medical diagnosis or high-stakes decision-making.

## 💼 Resume Summary

**AI-Powered Multimodal Emotion Analysis Platform**  
*Python, Streamlit, TensorFlow/Keras, PyTorch, Hugging Face Transformers, OpenCV, Librosa, SQLite*

- Developed a Streamlit application for analyzing emotion-related signals from facial images, audio, and text.
- Implemented separate modality pipelines and a late-fusion mechanism to combine available predictions.
- Integrated acoustic feature extraction, transformer-based text classification, and facial emotion classification.
- Added SQLite-based prediction history and automated tests for core components.

Use only the bullet points that accurately reflect the code you have verified and can explain in an interview.

## 📚 Additional Documentation

- [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md) — Detailed project explanation and interview preparation.
- `docs/ARCHITECTURE.md` — Architecture documentation, if present.
- `docs/MODEL_EVALUATION.md` — Evaluation details, if present.

---

**To run this project again later:** open PowerShell, navigate to the project folder, activate nothing manually if using the direct executable command above, run Streamlit, and open `http://localhost:8501`.
