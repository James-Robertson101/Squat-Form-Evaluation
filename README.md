# Squat Form Evaluation Tool

A web-based squat analysis system that evaluates squat form using computer vision. The application uses **MediaPipe pose estimation** and a Flask backend to process uploaded videos and return real-time feedback on squat technique.

---

## Features

- Upload squat videos via web interface
- Automatic repetition detection
- Joint-level form feedback (e.g. knee alignment, depth, back posture)
- Side and front view analysis modes
- Interactive results dashboard with per-rep breakdown
- Clean Flask + JavaScript AJAX integration (no page reload)

---

## How It Works

1. User uploads a squat video via the web interface  
2. Flask backend receives and stores the video  
3. Video is split into frames  
4. MediaPipe extracts human pose landmarks  
5. Trained evaluation model predicts Results
6. Results are returned as JSON  
7. Frontend (`feedback.js`) renders interactive feedback UI  

---

## Tech Stack

- **Backend:** Flask (Python)
- **Computer Vision:** MediaPipe, OpenCV
- **Frontend:** HTML, CSS, JavaScript
- **Data Handling:** JSON API via Fetch

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/James-Robertson101/Squat-Form-Evaluation.git
cd Squat-Form-Evaluation
```

### 2. Create a Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Running the app
To run the app:
```bash
python app.py
```

Then open in browser:
```bash 
http://127.0.0.1:5000
```