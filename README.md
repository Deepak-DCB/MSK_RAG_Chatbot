# MSK RAG Chatbot

A clinical-domain retrieval-augmented question-answering system built over the MSK Neurology corpus, integrating dense retrieval, heuristic biasing, and LLM-based reranking to produce context-aware, traceable answers.

**Live Demo (Streamlit Cloud):** *link to be added*  
**Demo Video:** *link to be added*

## Project Motivation

This project was originally built as a graduate capstone-style RAG system to interpret and answer questions within musculoskeletal neurology. It emphasizes:

- Vector-based dense retrieval over a custom domain corpus
- Heuristic and learned reranking of context candidates
- Query rewriting for domain specificity
- Token-budget-aware context packing
- Exposed telemetry for debugging and performance insight

## Architecture

1. Preprocessing and chunking of MSK article corpus  
2. Embedding generation with Sentence Transformers  
3. ChromaDB vector store persistence  
4. Retrieval + heuristic biasing  
5. Reranking via LLM  
6. Context packaging for answer generation  
7. Streamlit UI for interactive querying

## Setup and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/Deepak-DCB/MSK_RAG_Chatbot.git
   cd MSK_RAG_Chatbot

