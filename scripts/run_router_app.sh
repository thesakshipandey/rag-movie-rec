#!/bin/bash
# Launch script for Router-based RAG Movie Recommender Streamlit UI

cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec

# Add project root to PYTHONPATH so 'src' module can be found
export PYTHONPATH=/mnt/nas/sakshipandey/main/projects/rag-movie-rec:$PYTHONPATH

streamlit run src/app/router_app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --theme.base light


