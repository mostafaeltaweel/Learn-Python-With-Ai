"""
streamlit_app.py
------------------
واجهة المستخدم النهائية للمشروع (Streamlit UI) - نسخة مبسطة: بوكس سؤال فقط.

يقرأ التطبيق مفتاح Groq API من Streamlit Secrets عند النشر (لو الملف موجود فعلاً)،
أو من متغيرات البيئة (.env) عند التشغيل المحلي - ولا يحتوي أبدًا
على المفتاح الحقيقي مكتوبًا داخل الكود.
"""

import os
import streamlit as st
from importlib import import_module

st.set_page_config(
    page_title=" Assist Python ",
    page_icon="🐍",
    layout="centered",
)

rag = import_module("07_prompting")

# ------------------------------------------------------------------
# قراءة مفتاح الـ API: نحاول قراءة secrets.toml فقط لو موجود فعلاً على القرص،
# عشان نتفادى تحذير "No secrets found" لما نشتغل محليًا بدون هذا الملف.
# ------------------------------------------------------------------
_local_secrets_path = os.path.join(os.getcwd(), ".streamlit", "secrets.toml")
_has_secrets_file = os.path.exists(_local_secrets_path)

if _has_secrets_file or "STREAMLIT_SHARING_MODE" in os.environ or "STREAMLIT_RUNTIME" in os.environ:
    try:
        if not rag.GROQ_API_KEY:
            rag.GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
        rag.GROQ_MODEL = st.secrets.get("GROQ_MODEL", rag.GROQ_MODEL)
    except Exception:
        pass
else:
    # على Streamlit Cloud الملف مش موجود على القرص لكن secrets متاحة برضه،
    # فنحاول بأمان من غير ما نطبع تحذير محليًا.
    try:
        if hasattr(st, "secrets") and not rag.GROQ_API_KEY:
            rag.GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "") if _has_secrets_file else rag.GROQ_API_KEY
    except Exception:
        pass

DEFAULT_TOP_K = 4

# ------------------------------------------------------------------
# حالة المحادثة
# ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------------------------------------------------
# إدخال المستخدم - بوكس السؤال فقط
# ------------------------------------------------------------------
question = st.chat_input("اسأل عن أي كود موجود في المصادر...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not rag.GROQ_API_KEY:
            error_msg = (
                "لا يمكن توليد إجابة لأن مفتاح GROQ_API_KEY غير مُعرَّف. "
                "يرجى إضافته في ملف .env محليًا أو في Streamlit Secrets عند النشر."
            )
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            with st.spinner("جاري البحث في المصادر وتوليد الإجابة..."):
                try:
                    answer, sources = rag.answer_question(question, top_k=DEFAULT_TOP_K)
                except Exception as e:
                    answer = f"حدث خطأ أثناء توليد الإجابة: {e}"

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
