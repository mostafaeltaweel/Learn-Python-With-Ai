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

_retrieve_module = import_module("06_retrieve_context")
retrieve_context = _retrieve_module.retrieve_context
format_context = _retrieve_module.format_context

# لا تكتب المفتاح هنا أبدًا. القيمة الافتراضية فارغة عمدًا.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

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


def call_groq(prompt: str, api_key: str = None, model: str = None) -> str:
    """يستدعي نموذج Groq عبر SDK الرسمي الخاص بمكتبة groq."""
    api_key = api_key or GROQ_API_KEY
    model = model or GROQ_MODEL

    if not api_key:
        raise ValueError(
            "مفتاح GROQ_API_KEY غير موجود. "
            "أضِفه في ملف .env محليًا أو في Streamlit Secrets عند النشر."
        )

    from groq import Groq
    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def answer_question(question: str, top_k: int = 4, api_key: str = None, model: str = None):
    """الدالة الشاملة: استرجاع + بناء prompt + استدعاء LLM. تُستخدم من واجهة Streamlit."""
    retrieved_chunks = retrieve_context(question, top_k=top_k)
    context = format_context(retrieved_chunks)
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
            print(f"- {s.title} ({s.source})")
    except ValueError as e:
        print(e)
