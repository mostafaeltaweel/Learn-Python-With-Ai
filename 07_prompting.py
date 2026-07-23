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

# استيراد وحدة الاسترجاع ديناميكيًا لأن اسم الملف يبدأ برقم
_retrieve_module = import_module("06_retrieve_context")
retrieve_context = getattr(_retrieve_module, "retrieve_context", None)
format_context = getattr(_retrieve_module, "format_context", None)

# لا تكتب المفتاح هنا أبدًا. القيمة الافتراضية تُقرأ من البيئة
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = """أنت مساعد ذكي ومتخصص في شرح أكواد Python ومفاهيم التعلم الآلي وتحليل البيانات.
أجب على أسئلة المستخدم بالعربية فقط وبشكل واضع ومنظم، بالاعتماد حصريًا على السياق (Context) المُرفق أدناه.
إذا لم يحتوِ السياق على إجابة كافية للسؤال، صرّح بوضوح أن المعلومة غير متوفرة في المصادر المتاحة.
في نهاية إجابتك، اذكر دائمًا قائمة بعنوان "📖 المصادر المعتمدة:" تحتوي على أسماء وعناوين المصادر
التي اعتمدت عليها في إجابتك (من بين المصادر المرفقة في السياق فقط)."""


def build_prompt(question: str, context: str) -> str:
    return f"""السياق المسترجع من قاعدة المعرفة:
{context}

سؤال المستخدم:
{question}

اكتب إجابة واضحة ومنظمة بالعربية بناءً على السياق أعلاه فقط، ثم اذكر المصادر المستخدمة في النهاية."""


def call_groq(prompt: str, api_key: str = None, model: str = None) -> str:
    """يستدعي نموذج Groq عبر SDK الرسمي الخاص بمكتبة groq."""
    # جلب المفتاح الممرر أو المتغير العام أو متغير البيئة
    active_api_key = api_key or GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    active_model = model or GROQ_MODEL or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    if not active_api_key:
        raise ValueError(
            "مفتاح GROQ_API_KEY غير موجود. "
            "أضِفه في ملف .env محليًا أو في Streamlit Secrets عند النشر."
        )

    try:
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
    except Exception as e:
        raise RuntimeError(f"حدث خطأ أثناء الاتصال بـ Groq API: {str(e)}")


def answer_question(question: str, top_k: int = 4, api_key: str = None, model: str = None):
    """الدالة الشاملة: استرجاع + بناء prompt + استدعاء LLM. تُستخدم من واجهة Streamlit."""
    
    # 1. استرجاع القطع النصية
    if callable(retrieve_context):
        retrieved_chunks = retrieve_context(question, top_k=top_k)
    else:
        retrieved_chunks = []

    # 2. تنسيق السياق
    if callable(format_context):
        context = format_context(retrieved_chunks)
    else:
        # معالجة احتياطية في حال اختلاف شكل الـ Chunks
        context_list = []
        for c in retrieved_chunks:
            text = getattr(c, "page_content", getattr(c, "text", str(c)))
            context_list.append(text)
        context = "\n\n".join(context_list)

    # 3. بناء الـ Prompt واستدعاء Groq
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
            title = getattr(s, "title", s.metadata.get("source", "مستند") if hasattr(s, "metadata") else "مصدر")
            print(f"- {title}")
    except Exception as e:
        print("خطأ:", e)
