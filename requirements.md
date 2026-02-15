# Kirana Store Manager: Project Requirements

## 1. Mobile Framework (Kotlin/Android)

- **Min SDK:** 24 (Android 7.0)
- **Target SDK:** 34 (Android 14)

### Key Dependencies:
- `androidx.compose.ui`: Jetpack Compose for the UI
- `androidx.camera.camera2`: For scanning Khatabook pages
- `com.google.mlkit:text-recognition`: Local OCR for initial text extraction
- `com.squareup.retrofit2:retrofit`: To connect to the Python AI backend
- `androidx.room:room-runtime`: For local data caching

## 2. AI & Backend (Python)

These are the libraries required for your FastAPI and LangChain implementation. You can install these via `pip install -r requirements.txt`.

### Core API:
- `fastapi`
- `uvicorn`

### Database:
- `mysql-connector-python`
- `sqlalchemy`

### AI & RAG:
- `langchain`: Framework for LLM orchestration
- `langchain-community`: For Llama API and tool integrations
- `chromadb`: Vector database for storing government data/business studies
- `sentence-transformers`: For creating text embeddings

### Data Processing:
- `pandas`
- `openpyxl` (for Excel parsing)

### Scraping:
- `beautifulsoup4`
- `requests`

## 3. Infrastructure & Database

- **Database:** MySQL 8.0+
- **Environment:** Python 3.10+
- **LLM Access:** Llama 3.1 API Key (via Groq, Together AI, or local Ollama)

## 4. Functional Requirements Checklist

- [ ] OCR Pipeline: Convert image/photo of Khatabook to structured JSON
- [ ] Data Sync: Push Excel/OCR data into MySQL relational tables
- [ ] RAG System: Retrieve relevant GST rules and seasonal sales data
- [ ] Chat Interface: Kotlin-based UI to query the Llama-powered assistant
