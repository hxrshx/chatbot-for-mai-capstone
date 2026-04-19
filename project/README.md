# CAIRA - THWS MAI Chatbot

AI assistant for THWS Master of Artificial Intelligence students. Get answers about courses, admissions, housing, and campus life.

## Setup

**Requirements:**
- Python 3.13+
- 8GB RAM
- Tesseract OCR (for PDF processing)

**Quick Setup:**
```bash
# Clone and setup
git clone <your-repo-url>
cd mai-llm

# Create environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download AI model (~3GB)
bash scripts/download_qwen.sh

# Process documents and create knowledge base
python scripts/pre_processing.py
python scripts/generate_embeddings.py --input preprocessed/rag_chunks.jsonl --output preprocessed/embeddings
```

## Running CAIRA

**Start the chatbot:**
```bash
# Terminal 1: Backend
python backend/main.py

# Terminal 2: Frontend (new terminal)
python frontend/app.py
```

**Access:** http://localhost:7860

## What You Can Ask

- **Admissions**: "What are the admission requirements?"
- **Courses**: "What subjects are in the AI Master program?"
- **Housing**: "Where can I find accommodation in Würzburg?"
- **Campus**: "How do I register for exams?"
- **General**: "What is the semester ticket?"

## System Info

- 340 chunks from 59 THWS documents
- Qwen2.5-1.5B model (CPU optimized)
- German and English support
- ~30-60 second response time

## Troubleshooting

**Backend won't start:**
- Check if port 8000 is free: `lsof -ti:8000`
- Kill conflicting process: `kill <process_id>`

**No responses:**
- Ensure backend is running: `curl http://localhost:8000/health`
- Check frontend connects to backend

**Model download fails:**
- Check internet connection
- Ensure you have ~3GB free space

## Made for Students

This is a student project designed to help THWS MAI students navigate their academic journey. The chatbot contains official THWS documents and information relevant to Master of AI students.```

pip install -r requirements.txtUser → Gradio UI (Port 7860) → FastAPI Backend (Port 8000) → RAG System → Qwen LLM

```                                                                    ↓

                                                            FAISS Index (271 chunks)

### Running CAIRA```



**Start Backend** (in terminal 1):## ⚙️ Configuration

```bash

source venv/bin/activate### Speed Optimizations (Applied)

python backend/main.py- **Max tokens**: 150 (reduced from 256)

```- **Context size**: 600 chars per chunk (reduced from 1000)

- **Top-k chunks**: 2 (reduced from 3)

**Start Frontend** (in terminal 2):- **Repetition penalty**: 1.1

```bash

source venv/bin/activate  ### Performance

python frontend/app.py- **Local dev (CPU)**: 30-60 seconds per query

```- **Production (vLLM + GPU)**: 2-5 seconds per query



**Access CAIRA**: http://localhost:7860## 🎨 UI Features



## System Information- ✅ THWS branding (orange theme)

- ✅ University logo integration

- **Knowledge Base**: 323 chunks from 55 THWS MAI documents- ✅ Chat interface with history

- **Language Model**: Qwen2.5-1.5B-Instruct (CPU optimized)- ✅ Source document display

- **Embeddings**: multilingual-e5-small (German + English support)- ✅ Example questions

- **Response Time**: 30-60 seconds per query- ✅ Clean, student-friendly design

- **Features**: Cross-topic search, query expansion, ambiguity handling

## 📁 Project Structure

## Production Deployment

```

For production deployment with faster responses:mai-llm/

- Use GPU with vLLM for 2-5 second responses├── backend/

- Configure reverse proxy (nginx) for HTTPS│   └── main.py              # FastAPI backend

- Set environment variables for production settings├── frontend/

│   └── app.py               # Gradio UI

## License├── scripts/

│   ├── rag_with_llm.py      # RAG + LLM integration

Free for personal and educational use.│   └── rag_query.py         # RAG retrieval system

├── preprocessed/

---│   └── embeddings/          # FAISS index + chunks

© 2025-2026 CAIRA Team | THWS MAI Program├── models/
│   ├── qwen2.5-1.5b/        # Fast model (local dev)
│   └── qwen2.5-7b/          # Better quality (production)
└── Thws-logo_English.png    # University logo
```

## 🔧 Troubleshooting

### Backend won't start
```bash
# Check if port 8000 is in use
lsof -ti:8000 | xargs kill -9

# Restart backend
venv/bin/python backend/main.py
```

### Frontend shows "Backend not running"
1. Ensure backend is running (http://localhost:8000/health)
2. Check backend logs for errors
3. Restart both services

### Slow responses
- **Normal**: 30-60 seconds on CPU
- **To speed up**: Deploy with vLLM + GPU on server
- **Quick fix**: Reduce `max_new_tokens` further in `rag_with_llm.py`

## 🌐 Production Deployment

### For University Server

1. **Update configuration**:
   ```python
   # frontend/app.py
   API_URL = "http://your-server-ip:8000"
   ```

2. **Use vLLM for faster inference**:
   ```bash
   # On server
   python -m vllm.entrypoints.openai.api_server \
       --model models/qwen2.5-7b \
       --host 0.0.0.0 \
       --port 8000
   ```

3. **Update backend to use vLLM**:
   - Modify `rag_with_llm.py` to call vLLM API instead of loading model

4. **Setup reverse proxy** (nginx):
   ```nginx
   location /api {
       proxy_pass http://localhost:8000;
   }
   
   location / {
       proxy_pass http://localhost:7860;
   }
   ```

5. **Enable HTTPS** with Let's Encrypt

## 📝 API Endpoints

### Backend (FastAPI)
- `GET /` - Health check
- `GET /health` - Detailed status
- `POST /ask` - Ask question
- `GET /stats` - System statistics
- `GET /docs` - API documentation

### Example API Call
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I find housing?", "top_k": 2}'
```

## 🛠️ Development

### Install dependencies
```bash
pip install -r requirements_ui.txt
```

### Run tests
```bash
# Test backend
curl http://localhost:8000/health

# Test RAG system
python scripts/rag_with_llm.py --query "test question"
```

## 📚 Data Sources

- **Total chunks**: 271
- **Source documents**: 46 PDF/DOCX files
- **Topics**: admission, housing, courses, wurzburg-guide, general
- **Embedding model**: intfloat/multilingual-e5-small
- **LLM**: Qwen2.5-1.5B-Instruct (local), Qwen2.5-7B (production)

## 🎓 Credits

Built for THWS Master of Artificial Intelligence program
- RAG Technology
- Qwen 2.5 LLM
- FAISS Vector Search
- FastAPI + Gradio

## 📄 License

For educational use at THWS
