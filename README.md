# Karma Fraud Detector

A microservice for detecting fraudulent karma activities using machine learning.

## 🚀 Features
- Detects karma farming, mutual upvotes, and junk comments
- REST API with FastAPI
- Trained Random Forest model with TF-IDF for junk detection
- Configurable thresholds via `config.json`
- Dockerized for easy deployment

## 📁 Project Structure
```
.
├── main.py              # FastAPI app with endpoints
├── fraud_detector.py    # Feature extraction & per-activity scoring
├── train_model.py       # Train & save RandomForest model
├── config.json          # Thresholds, weights, and settings
├── model.pkl            # Trained model (output)
├── vectorizer.pkl       # Saved vectorizer (output)
├── Dockerfile           # Docker container setup
└── karma_fraud_dataset.json  # Training data (input)
```

## 🔧 Setup & Run

### 1. Train the Model
```bash
python train_model.py
```

### 2. Run Locally
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Visit docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Run with Docker
```bash
docker build -t karma-fraud-detector .
docker run -p 8000:8000 karma-fraud-detector
```

## 🧪 API Endpoints

### `POST /analyze`
**Input:**
```json
{
  "user_id": "stu_2293",
  "karma_log": [
    {
      "activity_id": "act_101",
      "type": "upvote_received",
      "from_user": "stu_3012",
      "timestamp": "2024-07-05T12:15:00Z",
      "source": "post",
      "post_id": "post_999"
    },
    {
      "activity_id": "act_102",
      "type": "comment",
      "content": "Nice post bro!",
      "timestamp": "2024-07-05T12:14:30Z"
    }
  ]
}
```

**Response:**
```json
{
  "user_id": "stu_2293",
  "fraud_score": 0.72,
  "suspicious_activities": [
    {
      "activity_id": "act_101",
      "reason": "Repeated upvotes from the same peer",
      "score": 0.81
    },
    {
      "activity_id": "act_102",
      "reason": "Low-effort comment flagged as karma bait",
      "score": 0.65
    }
  ],
  "status": "flagged"
}
```

### `GET /health`
Returns: `{ "status": "ok" }`

### `GET /version`
Returns: model and config version

## ✅ Testing
Use sample entries in `karma_fraud_dataset.json` or try your own logs.

## 📦 Deployment
Ensure `model.pkl`, `vectorizer.pkl`, and `config.json` are in the working directory before building the Docker container.

---

© 2025 Karma Fraud Detector
#   K a r m a _ F r a u d _ D e t e c t o r 
 
 