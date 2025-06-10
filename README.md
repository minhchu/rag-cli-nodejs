# Prerequisites

- NodeJs (>= 18)
- [Ollama](https://ollama.com/)
- [Qdrant](https://qdrant.tech/documentation/quickstart/)

## Installation

1. Install and start Ollama:
   - Download from: https://ollama.ai
   - Pull required models:
     ```bash
     ollama pull llama3.1
     ollama pull nomic-embed-text
     ```

2. Install and start Qdrant:
   
   Option A - Using Docker (Recommended):
   ```
   docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant
   ```

   Option B - Using Binary:
   - Download from: https://github.com/qdrant/qdrant/releases
   - Run: `./qdrant --config-path config/config.yaml`
   
3. Install Node.js dependencies:
   - `pnpm install`

## Usage

```bash
# Ingest a PDF
node index.js ingest ./document.pdf

# Query the system
node index.js query "What is the main topic of the document?"

# List ingested documents
node index.js list

# Clear database
node index.js clear
```

## Slide

https://slides.com/minhchu/xay-dung-ung-dung-rag
