# streamlit_app.py
# Fusion RAG Chatbot with Reciprocal Rank Fusion (RRF) and Hugging Face LLM

import streamlit as st
from pathlib import Path
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS



# ========== 🧩 Configuration ==========
st.set_page_config(page_title="Fusion RAG Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Fusion RAG Chatbot")

# Secrets and paths
HF_TOKEN = st.secrets["HF_TOKEN"]
APP_DIR = Path(__file__).resolve().parent
FAISS_DIR = APP_DIR / "faiss_index"
INDEX_NAME = "index"  # Must match when saving/loading

# Embedding and LLM models
EMBED_MODEL_NAME = "sentence-transformers/sentence-t5-large"
LLM_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"  # ✅ Real LLM (not embedding model)

# ========== 🔐 Load Resources (Cached) ==========

@st.cache_resource
def load_embeddings():
    """Load Hugging Face embeddings model."""
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": "cpu"}
    )
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/sentence-t5-large")
docs = [Document(page_content="...")]  # your documents
db = FAISS.from_documents(docs, embeddings)
db.save_local("faiss_index")  # Saves index.faiss and index.pkl

@st.cache_resource
def load_vector_store(_embeddings):
    """Load FAISS index from local directory with safety checks."""
    faiss_path = FAISS_DIR / f"{INDEX_NAME}.faiss"
    pkl_path = FAISS_DIR / f"{INDEX_NAME}.pkl"

    # Debug info
    st.write(f"📁 Checking FAISS index in: `{FAISS_DIR.absolute()}`")

    if not FAISS_DIR.exists():
        st.error("❌ FAISS directory not found. Make sure `faiss_index/` exists.")
        st.stop()

    if not faiss_path.exists():
        st.error(f"❌ FAISS index file not found: `{faiss_path}`")
        st.info("💡 Tip: Did you upload `index.faiss` to GitHub?")
        st.stop()

    if not pkl_path.exists():
        st.error(f"❌ Pickle file not found: `{pkl_path}`")
        st.info("💡 Both `.faiss` and `.pkl` files are required.")
        st.stop()

    try:
        db = FAISS.load_local(
            folder_path=str(FAISS_DIR),
            embeddings=_embeddings,
            index_name=INDEX_NAME,
            allow_dangerous_deserialization=True  # Required for newer LangChain
        )
        st.success("✅ FAISS index loaded successfully!")
        return db
    except Exception as e:
        st.error(f"❌ Failed to load FAISS index: {e}")
        st.stop()

@st.cache_resource
def get_llm_client():
    """Initialize Hugging Face Inference Client for text generation."""
    try:
        return InferenceClient(model=LLM_MODEL_NAME, token=HF_TOKEN)
    except Exception as e:
        st.error(f"❌ Failed to initialize LLM client: {e}")
        st.info("💡 Check if your HF token is valid and has access to the model.")
        st.stop()


# ========== 🚀 Load Resources ==========
embeddings = load_embeddings()
db = load_vector_store(embeddings)
retriever = db.as_retriever(search_kwargs={"k": 4})
client = get_llm_client()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ========== 🔍 Fusion RAG Logic ==========
def generate_queries(original_query: str):
    """Expand the original query into multiple related questions."""
    return [
        original_query,
        f"Explain in detail: {original_query}",
        f"What are the benefits of {original_query}?",
        f"What are the challenges or drawbacks of {original_query}?",
        f"Give a real-world application of {original_query}"
    ]

def reciprocal_rank_fusion(results_dict: dict, k: int = 60):
    """
    Fuse retrieval results using Reciprocal Rank Fusion (RRF).
    Higher score = more relevant.
    """
    fused_scores = {}
    for query, docs in results_dict.items():
        for rank, doc in enumerate(docs):
            content = doc.page_content.strip()
            fused_scores[content] = fused_scores.get(content, 0) + 1 / (rank + k)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

def fusion_rag_answer(query: str):
    """
    Full Fusion RAG pipeline:
    1. Expand query
    2. Retrieve for each
    3. RRF fusion
    4. Prompt LLM with context
    """
    # 1. Generate multiple queries
    expanded_queries = generate_queries(query)

    # 2. Retrieve documents for each query
    all_results = {q: retriever.get_relevant_documents(q) for q in expanded_queries}

    # 3. Fuse results using RRF
    reranked = reciprocal_rank_fusion(all_results)

    # 4. Build context from top 5 unique passages
    top_passages = [content for content, _ in reranked[:5]]
    context = "\n\n".join(top_passages)

    # 5. Create prompt
    prompt = f"""
Imagine you are chatting with me as my study buddy. 
I’ll give you some context, and you need to answer my question based on it.  

Here’s how I’d like you to reply:
- Stick only to the details from the context. 
- If the context doesn’t cover it, just say: 
  "The context does not provide this information."
- Write in a friendly, easy-to-follow way. 
- Feel free to break things into short bullets if it helps.

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
            temperature=0.2,
            do_sample=False,
            truncate=1024,
            stop_sequences=["\n\n", "Question:", "Context:"]
        )
    except Exception as e:
        return f"❌ LLM generation error: {str(e)}"

    # 7. Extract final answer
    raw_output = ""
    if isinstance(response, str):
        raw_output = response
    elif isinstance(response, dict):
        raw_output = response.get("generated_text", "")
    elif isinstance(response, list):
        raw_output = response[0].get("generated_text", "")

    # Extract after "Final Answer:"
    final_answer = raw_output.split("Final Answer:", 1)[-1].strip()
    return final_answer if final_answer else "I couldn't generate a proper answer."

# ========== 💬 UI: Chat Interface ==========
st.markdown("Ask me anything about your documents!")

query = st.text_input("🔍 Your question:", placeholder="E.g., What is quantum computing?")

if query:
    with st.spinner("🧠 Thinking... retrieving and fusing results..."):
        answer = fusion_rag_answer(query)
        st.session_state.chat_history.append({"question": query, "answer": answer})

# Display chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("🗨️ Conversation History")
    for i, chat in enumerate(st.session_state.chat_history, 1):
        st.markdown(f"**Q{i}:** {chat['question']}")
        st.markdown(f"**A{i}:** {chat['answer']}")
        st.markdown("---")
