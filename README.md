## RAG Model for Web Data and PDF Data Using Google Gemma

A hybrid Retrieval-Augmented Generation (RAG) system that:
- Scrapes the web with Google search, cleans and chunks the content
- Reads, cleans, and chunks uploaded PDFs
- Embeds both sources with `all-mpnet-base-v2` and stores in local pickle files
- Supports multiple retrieval strategies (dot product, cosine, Euclidean, FAISS, Hybrid BM25+Embeddings, Cross-Encoder reranking)
- Optionally generates answers with Google Gemma (2B/7B) using Hugging Face Transformers
- Exposes a Streamlit UI to orchestrate all workflows

### Highlights
- Web + PDF pipelines or either source alone
- Gemma-based LLM answer synthesis grounded in retrieved chunks
- Hash-based deduplication and incremental embedding updates
- Caching of scraped pages and local embedding storage
- Configurable retrieval algorithm and number of Google results
- Chapter 1 auto-detection for PDFs (or use full PDF)

## Quickstart

### 1) Environment
- Python 3.9–3.11 recommended
- Optional GPU with CUDA for best performance (CPU works but is slower; 4-bit quantization supported if `bitsandbytes` available)

### 2) Install dependencies
```bash
# From repo root
pip install --upgrade pip
pip install -r requirements.txt

# Optional: install PyTorch with CUDA (adjust for your CUDA version)
# See: https://pytorch.org/get-started/locally/

# Example from comments in requirements.txt (change versions if needed):
# pip install --upgrade --force-reinstall torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
```

Note:
- On Apple Silicon/macOS, you’ll likely run CPU or Metal acceleration for Transformers; `bitsandbytes` may not be available. The app will gracefully disable 4-bit quantization if unavailable.
- The first run will download models like `all-mpnet-base-v2`, cross-encoders, and Gemma (if used).

### 3) Run the app
```bash
streamlit run Streamlit_Driver.py
```

### 4) Use the UI
- Enter a query
- Choose number of Google results
- Optionally upload a PDF
- Pick a Search Mode and RAG method
- Click “Run Query”

```51:59:/Users/yash/Desktop/Yash/github/RAG-Model-For-Web-Data-and-PDF-Data-Using-Google-Gemma/Streamlit_Driver.py
search_mode = st.selectbox(
    "Choose Search Mode",
    ["Web_search_RAG", "PDF_Search_RAG", "Pdf_Google_search", "LLM_GoogleResults_PDF", "LLM"]
)

rag_search_type = st.selectbox(
    "Choose RAG Search Method",
    ["dot product", "cosine", "euclidean", "faiss", "hybrid_BM25_Embeddings", "cross_encoder"]
)
```

## Features

- **Web pipeline**:
  - Google search via `googlesearch-python`
  - HTML extraction with `readability-lxml` + `BeautifulSoup`
  - Aggressive boilerplate removal, language filtering (English), duplicate filtering by content hash
  - Sentence segmentation using spaCy sentencizer and chunking
- **PDF pipeline**:
  - Chapter 1 auto-detection to skip front matter or use full PDF
  - Page text extraction via PyMuPDF (`fitz`)
  - Sentence segmentation and chunking with overlap (PDFs use overlap; Web typically does not)
- **Embeddings**:
  - `SentenceTransformer("all-mpnet-base-v2")`
  - Incremental embedding with hash-based deduplication
  - Persisted to `.pkl` under `EmbeddingStorage/`
- **Retrieval**:
  - Dot Product, Cosine, Euclidean
  - FAISS (IVF/Flat as coded; both L2 and IP variants used in modules)
  - Hybrid BM25 + Embeddings (BM25 Okapi for lexical + dense similarity)
  - Cross-Encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **LLM Answering**:
  - Gemma 2B/7B (IT) via Hugging Face Transformers
  - Automatic FlashAttention 2 detection (fallback to SDPA)
  - Optional 4-bit quantization via `bitsandbytes`
  - Context-building from top retrieved chunks and answer generation

## Repository Structure

- `Streamlit_Driver.py`: Streamlit UI and mode routing
- `Web_RAG_Model.py`: Web-only RAG pipeline (`WEB_RAG_Application`)
- `Pdf_RAG_Model.py`: PDF-only RAG pipeline (`RAG_PDF_Application`)
- `PDF_Web_RAG_Model.py`: Combined PDF+Web retrieval (`WEB_PDF_RAG_Application`)
- `LLM_Pdf_Web_RAG.py`: Combined RAG + LLM answer pipeline (`WEB_PDF_LLM_RAG_Application`)
- `LLM_Module.py`: Standalone LLM querying (`LLM_Application`)
- `ImportsForModel.py`: Centralized imports, device/model config, quantization defaults
- `EmbeddingStorage/`: Created at runtime, stores `*.pkl` embeddings
- `scraped_texts/`, `cache/`: Created at runtime for scraped content and metadata

## Search Modes (UI)

- **Web_search_RAG**: Run the web pipeline, embed, and retrieve chunks from web pages only.
- **PDF_Search_RAG**: Parse and chunk the uploaded PDF and retrieve from PDF chunks only.
- **Pdf_Google_search**: Combined PDF + Web pipeline with unified retrieval.
- **LLM_GoogleResults_PDF**: Combined RAG → retrieve top chunks → feed into Gemma for an answer; returns both final answer and supporting chunks.
- **LLM**: Direct LLM call with your query (no RAG grounding).

Returned results are rendered in the UI. For LLM + RAG mode, you’ll see an answer plus top supporting chunks:
```154:179:/Users/yash/Desktop/Yash/github/RAG-Model-For-Web-Data-and-PDF-Data-Using-Google-Gemma/Streamlit_Driver.py
if "answer" in results and "top_chunks" in results:
    st.markdown("## 🧾 Final Answer")
    st.markdown(f"> {results['answer']}")
    st.markdown("## 🧠 Top Supporting Chunks")
    for i, res in enumerate(results["top_chunks"]):
        ...
```

## Retrieval Algorithms

All modules normalize or map retrieval names consistently; combined pipeline selects based on `rag_search_type`:

```553:602:/Users/yash/Desktop/Yash/github/RAG-Model-For-Web-Data-and-PDF-Data-Using-Google-Gemma/PDF_Web_RAG_Model.py
rag_type = (self.rag_search_type or "dot product").lower().strip()
if rag_type in ["dot product", "cosine"]:
    ...
elif rag_type == "euclidean":
    ...
elif rag_type == "faiss":
    ...
elif rag_type in ["hybrid", "bm25"]:
    ...
elif rag_type in ["cross", "cross encoder", "cross-encoder"]:
    ...
```

Available methods:
- Dot Product or Cosine similarity using `sentence-transformers` utilities
- Euclidean distance with `sklearn.metrics.pairwise.euclidean_distances` or vectorized PyTorch
- FAISS ANN search (FlatIP/L2)
- BM25 + embedding rerank hybrid
- Cross-Encoder reranking (`ms-marco-MiniLM-L-6-v2`)

## LLM: Google Gemma Integration

### Models
Default model set centrally:
```128:131:/Users/yash/Desktop/Yash/github/RAG-Model-For-Web-Data-and-PDF-Data-Using-Google-Gemma/ImportsForModel.py
# LLM
model_id = "google/gemma-7b-it"
quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                         bnb_4bit_compute_dtype=torch.float16)
```

Each LLM-enabled module will:
- Choose attention implementation (FlashAttention 2 if available and compute capability allows, else SDPA)
- Configure quantization if `bitsandbytes` is installed
- Load the tokenizer and model from Hugging Face
- Build a prompt from the retrieved context (for RAG+LLM mode)

VRAM-aware adjustments:
- `WEB_PDF_LLM_RAG_Application.quantization_configuration_setup()` suggests Gemma 2B on low VRAM and 7B on higher VRAM; it toggles 4-bit where appropriate.
- If quantization isn’t available, models are loaded in float16 and moved to GPU if present.

If you need model access that requires auth (some Gemma weights), log in:
```bash
huggingface-cli login
```

## Data Flow and Storage

- Web scraping:
  - Google search → requests/Readability/BS4 clean → language filter (English) → deduplication via content hash
  - Saves minimal cache per-URL to `cache/` and raw text to `scraped_texts/` (in `Web_RAG_Model.py`)
- PDF ingestion:
  - PyMuPDF page text → chapter 1 auto-detection to skip TOC/front matter → sentence segmentation and chunking
- Chunking:
  - Web: typically non-overlapping, length-based or sentence-group chunking
  - PDF: overlapping sentence groups to preserve context
- Embedding:
  - Uses `all-mpnet-base-v2` on GPU/CPU
  - Dedup by hash and append-only updates to `.pkl` files
- Storage:
  - Web-only: `EmbeddingStorage/WebLinks_EmbeddedData.pkl`
  - PDF-only: `EmbeddingStorage/PDF_EmbeddedData.pkl`
  - Combined (RAG+LLM): `EmbeddingStorage/EmbeddedData.pkl`

## Programmatic Usage

### Web-only RAG
```python
from Web_RAG_Model import WEB_RAG_Application

app = WEB_RAG_Application(topic="insulin resistance", number_results=5, mode="Web_search_RAG", verbose=False)
app.run_web_pipeline()
results, method = app.Semantic_RAG_Search(query="How to reverse insulin resistance?", rag_search_type="cosine")
print(method, results[:2])
```

### PDF-only RAG
```python
from Pdf_RAG_Model import RAG_PDF_Application

with open("your.pdf", "rb") as f:
    pdf_bytes = f.read()

app = RAG_PDF_Application(topic="neural networks", number_results=0, mode="PDF_Search_RAG", pdf_bytes=pdf_bytes)
app.run_pdf_pipeline()
results, method = app.Semantic_Rag_DotProduct_Search(query="What is backpropagation?", rag_search_type="dot product")
print(method, results[:2])
```

### Combined PDF + Web RAG
```python
from PDF_Web_RAG_Model import WEB_PDF_RAG_Application

with open("your.pdf", "rb") as f:
    pdf_bytes = f.read()

app = WEB_PDF_RAG_Application(
    topic="attention mechanisms",
    number_results=5,
    mode="Pdf_Google_search",
    pdf_bytes=pdf_bytes,
    rag_search_type="faiss",
    file_name="your.pdf"
)
results, method = app.Data_Gathering_Processing(rag_search_type="faiss")
print(method, results)
```

### LLM over Combined RAG
```python
from LLM_Pdf_Web_RAG import WEB_PDF_LLM_RAG_Application

with open("your.pdf", "rb") as f:
    pdf_bytes = f.read()

app = WEB_PDF_LLM_RAG_Application(
    topic="What are the risks of intermittent fasting?",
    number_results=5,
    mode="LLM_GoogleResults_PDF",
    pdf_bytes=pdf_bytes,
    file_name="your.pdf"
)
app.Data_Gathering_Processing()
app.LLM_Model_Setup()
answer_bundle = app.LLM_PDF_WEB_Query_Search(query="Is IF safe for diabetics?")
print(answer_bundle["answer"])
```

### Direct LLM (no RAG)
```python
from LLM_Module import LLM_Application

app = LLM_Application(topic="Summarize transformers attention", number_results=0, mode="LLM")
text, method = app.SearchModuleSetup_LLM()  # returns raw model text and "LLM"
print(text)
```

## Configuration

- UI knobs:
  - Query, number of URLs, snippet length, PDF upload, “Use entire PDF”, Search Mode, RAG method, Verbose output
- Retrieval algorithms:
  - `dot product`, `cosine`, `euclidean`, `faiss`, `hybrid_BM25_Embeddings`, `cross_encoder`
- Hardware:
  - GPU detection, FlashAttention 2 where available, `bitsandbytes` 4-bit quantization if present

## Requirements

Key dependencies in `requirements.txt`:
```text
langchain
readability-lxml
beautifulsoup4
sentence-transformers
googlesearch-python
tqdm
nltk
langdetect
pandas
rank-bm25
streamlit
PyMuPDF
spacy
matplotlib
faiss-cpu
huggingface
accelerate
bitsandbytes
```

Notes:
- NLTK `punkt` is downloaded at import time.
- spaCy uses the lightweight `English` sentencizer; no large model download required.
- If you have a CUDA GPU, prefer installing `faiss-gpu` and a CUDA-enabled PyTorch.

## Troubleshooting

- Torch/Streamlit crash at startup:
  - The app proactively patches Streamlit’s file watcher to ignore `torch.classes`, preventing crashes.
- bitsandbytes not available:
  - The code disables quantization and runs the model in float16; expect higher VRAM usage.
- “No results returned”:
  - Check that embeddings were saved and `.pkl` files exist in `EmbeddingStorage/`
  - Increase Google results; adjust query specificity
- Web scraping issues:
  - Some sites block bots; results skip those pages
  - Non-English pages are filtered; try different queries if you expect multi-lingual sources
- PDF parsing:
  - Chapter detection is heuristic; use “Use entire PDF” if detection misses content

## Ethics and Legal

- Respect robots.txt and site terms of service when scraping.
- Only use Gemma and other models under their respective licenses and terms.