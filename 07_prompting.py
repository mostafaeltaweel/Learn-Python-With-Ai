"""
07_prompting.py (النسخة المتقدمة المحسنة)
--------------------------------------
المرحلة السابعة: بناء Prompt احترافي واستدلال عالي الدقة عبر Groq API.
"""

import os
from importlib import import_module

# 1. تحميل متغيرات البيئة
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 2. استيراد وحدة الاسترجاع
try:
    _retrieve_module = import_module("06_retrieve_context")
    retrieve_context = getattr(_retrieve_module, "retrieve_context", None)
    format_context = getattr(_retrieve_module, "format_context", None)
except Exception:
    retrieve_context = None
    format_context = None

# ------------------------------------------------------------------
# المتغيرات العامة الأساسية
# ------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# توجيهات صارمة ومحددة للنموذج لرفع جودة الإجابة
SYSTEM_PROMPT = """أنت خبير ذكاء اصطناعي وتعلّم عميق، ودورك شرح وتفسير أكواد Python المرفقة في السياق فقط.

قواعد الاستجابة الصارمة:
1. الإجابة يجب أن تكون باللغة العربية، واضحة، ومقسمة إلى نقاط أو خطوات منظمة.
2. اعتمد حصريًا على السياق (Context) المرفق لك في السؤال. لا تقم بافتراض أو اختراع أي معلومات خارجية.
3. إذا لم يغطّ السياق سؤال المستخدم بشكل كامل، اذكر بوضوح: "المعلومة المطلوبة غير متوفرة بشكل كامل في المصادر المتاحة."
4. قم بشرح الأكواد والخطوات بالتفصيل بناءً على ما هو موجود في المصدر.
5. في نهاية الإجابة، قم بإدراج قسم خاص بعنوان "المصادر المعتمدة:" واذكر فيه ألقاب وأسماء الملفات التي استندت إليها فقط."""


def build_prompt(question: str, context: str) -> str:
    """يبني نص الـ Prompt الموجه للنموذج بشكل معالج."""
    if not context.strip():
        context = "لا يوجد سياق مسترجع متاح."

    return f"""### السياق المسترجع من قاعدة المعرفة:
{context}

---
### سؤال المستخدم:
{question}

اكتب إجابة دقيقة ومفصلة بناءً على السياق أعلاه فقط:"""


def call_groq(prompt: str, api_key: str = None, model: str = None, temperature: float = 0.1) -> str:
    """استدعاء مباشر ومُحسّن لـ Groq API مع ضبط المعلمات لتفادي الهلوسة."""
    active_api_key = api_key or GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    active_model = model or GROQ_MODEL or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    if not active_api_key:
        raise ValueError("مفتاح GROQ_API_KEY غير متوفر. يرجى إضافته في البيئة أو الملفات.")

    try:
        from groq import Groq
        client = Groq(api_key=active_api_key)

        response = client.chat.completions.create(
            model=active_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,  # منخفضة للالتزام بالحقائق
            max_tokens=2048,
            top_p=0.9,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"خطأ في الاتصال بـ Groq API: {str(e)}")


def get_groq_llm(api_key: str = None, model_name: str = None, temperature: float = 0.1):
    """إرجاع كائن استدعاء متوافق مع سلاسل LangChain وواجهة Streamlit."""
    active_api_key = api_key or GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    active_model = model_name or GROQ_MODEL or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    try:
        from langchain_groq import ChatGroq
        return ChatGroq(
            groq_api_key=active_api_key,
            model_name=active_model,
            temperature=temperature,
        )
    except Exception:
        class GroqLLMWrapper:
            def __init__(self, key, model, temp):
                self.key = key
                self.model = model
                self.temp = temp

            def invoke(self, prompt):
                p_text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
                res = call_groq(p_text, api_key=self.key, model=self.model, temperature=self.temp)
                class Response:
                    content = res
                return Response()

            def __call__(self, prompt):
                return self.invoke(prompt)

        return GroqLLMWrapper(active_api_key, active_model, temperature)


def build_rag_chain(retriever, llm):
    """بناء RAG Chain متكامل ومستقر للواجهة."""
    class RAGChain:
        def __init__(self, retriever_obj, llm_obj):
            self.retriever = retriever_obj
            self.llm = llm_obj

        def invoke(self, inputs: dict) -> dict:
            question = inputs.get("input", "") if isinstance(inputs, dict) else str(inputs)

            # 1. الاسترجاع
            if hasattr(self.retriever, "invoke"):
                docs = self.retriever.invoke(question)
            elif hasattr(self.retriever, "get_relevant_documents"):
                docs = self.retriever.get_relevant_documents(question)
            elif callable(self.retriever):
                docs = self.retriever(question)
            elif callable(retrieve_context):
                docs = retrieve_context(question)
            else:
                docs = []

            # 2. التنسيق
            if callable(format_context):
                context_str = format_context(docs)
            else:
                parts = []
                for i, doc in enumerate(docs, 1):
                    text = getattr(doc, "page_content", getattr(doc, "text", str(doc)))
                    source = doc.metadata.get("source", f"مصدر {i}") if hasattr(doc, "metadata") else f"مصدر {i}"
                    parts.append(f"[{source}]\n{text}")
                context_str = "\n\n---\n\n".join(parts)

            # 3. التوليد
            full_prompt = build_prompt(question, context_str)
            
            if hasattr(self.llm, "invoke"):
                response = self.llm.invoke(full_prompt)
                answer = getattr(response, "content", str(response))
            else:
                answer = call_groq(full_prompt)

            return {
                "answer": answer,
                "context": docs
            }

    return RAGChain(retriever, llm)
