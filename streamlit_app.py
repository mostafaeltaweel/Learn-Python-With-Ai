"""
streamlit_app.py
------------------
واجهة المستخدم النهائية (Streamlit UI) المربوطة بـ Groq API
"""

import os
import streamlit as st
from importlib import import_module

# 1. إعداد الصفحة
st.set_page_config(
    page_title="Data Analysis RAG Mentor",
    page_icon="🐍",
    layout="centered"
)

# 2. استيراد الوحدات المرقّمة
doc_m = import_module("01_documents")
prep_m = import_module("02_preprocessing")
chunk_m = import_module("03_chunking")
embed_m = import_module("04_vector_representation")
store_m = import_module("05_create_chroma_store")
ret_m = import_module("06_retrieve_context")
prompt_m = import_module("07_prompting")

# 3. قراءة مفتاح Groq من البيئة المحلية أو Streamlit Secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

try:
    if hasattr(st, "secrets") and not GROQ_API_KEY:
        GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    pass

# 4. بناء وتحميل السلسلة في الذاكرة المؤقتة
@st.cache_resource(show_spinner="⏳ جاري إعداد قاعدة البيانات ونموذج Groq...")
def load_rag_pipeline(api_key: str):
    embeddings = embed_m.get_embedding_model()
    
    # بناء تلقائي لقاعدة البيانات إذا لم تكن موجودة على السحابة
    chroma_dir = "./chroma_db"
    if not os.path.exists(chroma_dir) or not os.listdir(chroma_dir):
        docs = doc_m.load_documents()
        cleaned = prep_m.preprocess_documents(docs)
        chunks = chunk_m.chunk_documents(cleaned)
        vector_store = store_m.create_or_load_vector_store(
            chunks=chunks, 
            embedding_model=embeddings,
            persist_directory=chroma_dir
        )
    else:
        vector_store = store_m.create_or_load_vector_store(
            embedding_model=embeddings,
            persist_directory=chroma_dir
        )

    retriever = ret_m.get_retriever(vector_store)
    llm = prompt_m.get_groq_llm(api_key=api_key)
    chain = prompt_m.build_rag_chain(retriever, llm)
    return chain

# 5. إدارة حالة المحادثة
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🐍 Data Analysis RAG Mentor")
st.caption("Powered by Groq (Llama 3.3) + ChromaDB")

# عرض المحادثات السابقة
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📖 المصادر المعتمدة"):
                for src in msg["sources"]:
                    st.markdown(f"- {src}")

# 6. صندوق إدخال السؤال
question = st.chat_input("اسأل عن أي كود أو مفهوم في تحليل البيانات...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not GROQ_API_KEY:
            error_msg = "❌ مفتاح GROQ_API_KEY غير موجود! يرجى إضافته في Secrets أو ملف .env"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})
        else:
            with st.spinner("جاري البحث في المصادر وتوليد الإجابة..."):
                try:
                    chain = load_rag_pipeline(GROQ_API_KEY)
                    result = chain.invoke({"input": question.strip()})
                    
                    answer = result.get("answer", "لم يتم توليد إجابة.")
                    source_docs = result.get("context", [])
                    sources = ret_m.format_sources(source_docs)
                except Exception as e:
                    answer = f"حدث خطأ أثناء التوليد: {str(e)}"
                    sources = []

            st.markdown(answer)
            if sources:
                with st.expander("📖 المصادر المعتمدة", expanded=False):
                    for src in sources:
                        st.markdown(f"- {src}")

            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": sources
            })
