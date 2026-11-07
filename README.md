Enterprise AI Assistant

Project Overview:
This project is an advanced, modular Enterprise AI Assistant built using (Python, Flask, and Azure OpenAI). It is designed to handle complex user queries through an Adaptive RAG architecture, Agentic tool calling, and Multimodal image understanding. 

This repository serves as the final technical assessment for the AI Engineer role.
Key Features & Capabilities
1. Advanced Adaptive RAG: Implements Query Understanding, Routing, Adaptive Retrieval, and LLM-based Re-ranking to provide highly accurate, grounded answers with citations.
2. Agentic Tool Calling:Equipped with intelligent tools (Knowledge Base Search, Calculator, and Mock Ticket Creation) to take autonomous actions based on user intent.
3. Multimodal Image Understanding & OCR:Integrates Azure OpenAI Vision to process images, perform OCR for both Arabic and English, and answer visual questions.
4. Voice RAG (Bilingual):Features Azure Speech SDK integration for real-time Text-to-Speech (TTS) with automatic language detection (Arabic/English).
5. Conversation Memory:** Maintains multi-turn conversation history for context-aware interactions.
6. Automated Evaluation (LLM-as-a-Judge):Includes a custom evaluation script to measure faithfulness (hallucination check) and answer relevancy.

Project Architecture (Microservices-like)
The system is divided into modular components for clean architecture and scalability:
* `app.py`: The main Flask orchestrator and router (handles UI requests, voice synthesis, and image routing).
* `advanced_rag.py`: The core Adaptive RAG pipeline (Query rewriting, re-ranking, and sufficiency checks).
* `agent_tools.py`: Defines the executable agentic tools (Search, Calculator, Ticketing).
* `evaluate_bot.py`: A standalone evaluation script for performance testing.
* `templates/index.html`: The frontend user interface.

Prerequisites & Setup
To run this project locally, you need the following environment variables configured in a `.env` file:
```text
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_CHAT_DEPLOYMENT=your_deployment
AZURE_SEARCH_ENDPOINT=your_search_endpoint
AZURE_SEARCH_INDEX=your_index
AZURE_SEARCH_KEY=your_search_key
AZURE_SPEECH_KEY=your_speech_key
AZURE_SPEECH_REGION=your_speech_region