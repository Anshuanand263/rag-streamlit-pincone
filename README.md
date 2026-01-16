# rag-streamlit-pincone
<br>
<b>Retrieval-Augmented Generation (RAG) — Streamlit front-end + Pinecone vector store + Gemini embeddings & LLMs</b>

<p>A compact, student-friendly RAG demo that ingests documents (PDF / text), creates embeddings, stores vectors in Pinecone, and serves an interactive Streamlit question-answering UI with source attribution</p>
<hr>
<h2> Features</h2>
<ul>
    <li>Semantic search using Pinecone vector store</li>
    <li>Query rewriting using Gemini for improved retrieval</li>
    <li>Context-aware answers using RAG architecture</li>
    <li>Interactive chat UI built with Streamlit</li>
    <li>Session-based conversation memory</li>
    <li>Debug view for rewritten queries and retrieved context</li>
</ul>
<hr>
<h2> Tech Stack</h2>
<ul>
    <li><b>Frontend:</b> Streamlit</li>
    <li><b>LLM:</b> Google Gemini (gemini-2.5-flash)</li>
    <li><b>Embeddings:</b> Gemini Embedding Model</li>
    <li><b>Vector Database:</b> Pinecone</li>
    <li><b>Framework:</b> LangChain</li>
</ul>