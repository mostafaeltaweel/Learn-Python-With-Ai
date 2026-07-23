"""
07_prompting.py
-----------------
المرحلة السابعة: بناء الـ Prompt واستدعاء نموذج اللغة الكبير (LLM) عبر Groq API.

قواعد أمان مفتاح الـ API:
- لا يُكتب المفتاح الحقيقي داخل الكود إطلاقًا.
- يُقرأ المفتاح من متغير بيئة GROQ_API_KEY (محليًا عبر ملف .env)
  أو من Streamlit Secrets عند النشر (يتم ذلك من داخل streamlit_app.py).
"""

import os
from importlib import import_module

try:
    from dotenv import load_dotenv
    load_dotenv()  # يحمّل GROQ_API_KEY و GROQ_MODEL من ملف .env المحلي إن وُجد
except ImportError:
    pass

# استيراد وحدة الاسترجاع ديناميكيًا
_retrieve_module = import_module("06_retrieve_context")
retrieve_context = getattr(_retrieve_module, "retrieve_context", None)
format_context = getattr(_retrieve_module, "format_context", None)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = """أنت مساعد ذكي متخصص في شرح أكواد Python الخاصة بالتعلم الآلي والتعلم العميق.
أجب على أسئلة المستخدم بالعربية فقط، بالاعتماد حصريًا على السياق (Context) المُرفق أدناه.
إذا لم يحتوِ السياق على إجابة كافية للسؤال، صرّح بوضوح أن المعلومة غير متوفرة في المصادر المتاحة.
في نهاية إجابتك، اذكر دائمًا قائمة بعنوان "المصادر:" تحتوي على أسماء وعناوين المصادر
التي اعتمدت عليها في إجابتك (من بين المصادر المرفقة في السياق فقط)."""


def build_prompt(question: str, context: str) -> str:
    return f"""السياق المسترجع من قاعدة المعرفة:
{context}

سؤال المستخدم:
{question}

اكتب إجابة واضحة ومنظمة بالعربية بناءً على السياق أعلاه فقط، ثم اذكر المصادر المستخدمة في النهاية."""


def get_groq_llm(api_key: str = None, model_name: str = None, temperature: float = 0.2):
    """
    تنشئ وترجع كائن ChatGroq أو كائن متوافق لاستخدامه في سلاسل RAG والواجهة.
    """
    active_api_key = api_key or GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    active_model = model_name or GROQ_MODEL or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    if not active_api_key:
        raise ValueError(
            "مفتاح GROQ_API_KEY غير موجود. "
            "أضِفه في ملف .env محليًا أو في Streamlit Secrets عند النشر."
        )

    try:
        from langchain_groq import ChatGroq
        return ChatGroq(
            groq_api_key=active_api_key,
            model_name=active_model,
            temperature=temperature,
        )
    except ImportError:
        class GroqLLMWrapper:
            def __init__(self, key, model, temp):
                self.key = key
                self.model = model
                self.temp = temp

            def invoke(self, prompt):
                p_text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
                res = call_groq(p_text, api_key=self.key, model=self.model)
                class Response:
                    content = res
                return Response()

            def __call__(self, prompt):
                return self.invoke(prompt)

        return GroqLLMWrapper(active_api_key, active_model, temperature)


def call_groq(prompt: str, api_key: str = None, model: str = None) -> str:
    """يستدعي نموذج Groq عبر SDK الرسمي الخاص بمكتبة groq."""
    active_api_key = api_key or GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    active_model = model or GROQ_MODEL or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    if not active_api_key:
        raise ValueError(
            "مفتاح GROQ_API_KEY غير موجود. "
            "أضِفه في ملف .env محليًا أو في Streamlit Secrets عند النشر."
        )

    from groq import Groq
    client = Groq(api_key=active_api_key)

    response = client.chat.completions.create(
        model=active_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=2048,
    )
    return response.choices[0].message.content


def build_rag_chain(retriever, llm):
    """
    الدالة المطلوبة لـ streamlit_app.py:
    تبني وتسترجع سلسلة RAG متكاملة تقوم بـ (الاسترجاع ➔ بناء السياق ➔ التوليد).
    """
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

            # 2. تنسيق السياق
            if callable(format_context):
                context_str = format_context(docs)
            else:
                parts = []
                for i, doc in enumerate(docs, 1):
                    text = getattr(doc, "page_content", getattr(doc, "text", str(doc)))
                    parts.append(f"[مصدر {i}]\n{text}")
                context_str = "\n\n---\n\n".join(parts)

            # 3. بناء الـ Prompt والتوليد عبر LLM
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


def answer_question(question: str, top_k: int = 4, api_key: str = None, model: str = None):
    """الدالة الشاملة: استرجاع + بناء prompt + استدعاء LLM. تُستخدم من واجهة Streamlit."""
    retrieved_chunks = retrieve_context(question, top_k=top_k) if callable(retrieve_context) else []
    context = format_context(retrieved_chunks) if callable(format_context) else ""
    prompt = build_prompt(question, context)
    answer = call_groq(prompt, api_key=api_key, model=model)
    return answer, retrieved_chunks


if __name__ == "__main__":
    q = "اشرح لي فكرة كود الـ Autoencoder وإزالة التشويش من الصور"
    try:
        answer, sources = answer_question(q)
        print("الإجابة:\n", answer)
        print("\nالمصادر المسترجعة:")
        for s in sources:
            title = getattr(s, "title", "مصدر")
            source = getattr(s, "source", "")
            print(f"- {title} ({source})")
    except ValueError as e:
        print(e)
