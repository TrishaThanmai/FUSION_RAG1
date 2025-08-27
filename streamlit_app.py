# streamlit_app.py
# 🚀 Fusion RAG Chatbot with Reciprocal Rank Fusion (RRF)

import streamlit as st
from pathlib import Path
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ========== 🧩 Config ==========
st.set_page_config(page_title="Fusion RAG Chatbot", page_icon="🔍", layout="centered")
st.title("🔍 Fusion RAG Chatbot with RRF")

# Secrets and paths
HF_TOKEN = st.secrets["HF_TOKEN"]
APP_DIR = Path(__file__).resolve().parent
FAISS_DIR = APP_DIR / "faiss_index"
INDEX_NAME = "index"  # Must match saved index

# Models
EMBED_MODEL_NAME = "sentence-transformers/sentence-t5-large"
LLM_MODEL_NAME = "google/gemma-2-9b"  # Or "HuggingFaceH4/zephyr-7b-beta"

# ========== 🔐 Load Resources (Cached) ==========

@st.cache_resource
def load_embeddings():
    """Load embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": "cpu"}
    )

@st.cache_resource
def load_vector_store(_embeddings):
    """Load FAISS index with safety checks."""
    st.write(f"📁 Loading FAISS from: `{FAISS_DIR}`")

    if not FAISS_DIR.exists():
        st.error("❌ FAISS directory not found!")
        st.stop()

    try:
        db = FAISS.load_local(
            folder_path=str(FAISS_DIR),
            embeddings=_embeddings,
            index_name=INDEX_NAME,
            allow_dangerous_deserialization=True  # Required for pickle
        )
        st.success("✅ FAISS index loaded!")
        return db
    except Exception as e:
        st.error(f"❌ Failed to load FAISS: {e}")
        st.exception(e)
        st.stop()

@st.cache_resource
def get_llm_client():
    """Initialize Hugging Face Inference Client."""
    try:
        return InferenceClient(model=LLM_MODEL_NAME, token=HF_TOKEN)
    except Exception as e:
        st.error(f"❌ LLM Client Error: {e}")
        st.stop()

# ========== 🚀 Load Resources ==========
embeddings = load_embeddings()
db = load_vector_store(embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})
client = get_llm_client()

# ========== 🔍 Fusion RAG Logic ==========

def generate_queries(original_query: str):
    """Generate semantically related queries to improve recall."""
    return [
        original_query,
        f"Explain in detail: {original_query}",
        f"What are the key points about {original_query}?",
        f"How does {original_query} work?",
        f"Real-world applications of {original_query}"
    ]

def reciprocal_rank_fusion(results_dict: dict, k: int = 60):
    """
    Fuse ranked results using Reciprocal Rank Fusion.
    Higher score = higher relevance.
    """
    fused_scores = {}
    for query, docs in results_dict.items():
        for rank, doc in enumerate(docs):
            content = doc.page_content.strip()
            fused_scores[content] = fused_scores.get(content, 0) + 1 / (rank + k)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

def fusion_rag_answer(query: str):
    """Full Fusion RAG pipeline."""
    # 1. Expand query
    expanded_queries = generate_queries(query)

    # 2. Retrieve for each query
    all_results = {}
    for q in expanded_queries:
        try:
            docs = retriever.invoke(q)
            all_results[q] = docs
        except Exception as e:
            st.warning(f"⚠️ Retrieval failed for '{q}': {e}")

    if not all_results:
        return "I couldn't retrieve any relevant documents."

    # 3. RRF Fusion
    reranked = reciprocal_rank_fusion(all_results)

    # 4. Build context from top 5 unique passages
    top_passages = [content for content, _ in reranked[:5]]
    context = "\n\n".join(top_passages)

    # 5. Prompt LLM
    prompt = f"""
You are a helpful and precise assistant for question-answering tasks. 
Use only the following context to answer the question. 
Do not use prior knowledge or guess.

If the context does not contain the answer, respond exactly:
"The context does not provide this information."

Be concise and limit your answer to 2-3 sentences unless more detail is needed.."


Context:
{context}

Question: {query}

Final Answer:
"""

    # 6. Generate response
    try:
        response = client.text_generation(
            prompt,
            max_new_tokens=350,
            temperature=0.3,
            do_sample=True,
            stop_sequences=["\n\n", "Question:", "Context:", "Final Answer:"]
        )
    except Exception as e:
        return f"❌ LLM Error: {str(e)}"

    # 7. Extract answer
    raw_output = ""
    if isinstance(response, str):
        raw_output = response
    elif isinstance(response, dict):
        raw_output = response.get("generated_text", "")
    elif isinstance(response, list) and response:
        raw_output = response[0].get("generated_text", "")

    # Extract after "Final Answer:"
    final_answer = raw_output.split("Final Answer:", 1)[-1].strip()
    return final_answer if final_answer else "I don't know."

# ========== 💬 UI ==========
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown("Ask me anything about your documents!")

query = st.text_input("🔍 Your question:", placeholder="E.g., What is quantum entanglement?")

if query:
    with st.spinner("🧠 Expanding query, retrieving, and fusing results..."):
        answer = fusion_rag_answer(query)
        st.session_state.chat_history.append({"question": query, "answer": answer})

# Display chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("🗨️ Chat History")
    for i, chat in enumerate(st.session_state.chat_history, 1):
        st.markdown(f"**Q{i}:** {chat['question']}")
        st.markdown(f"**A{i}:** {chat['answer']}")
        st.markdown("---")
