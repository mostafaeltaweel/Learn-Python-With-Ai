"""
01_documents.py
----------------
المرحلة الأولى من مشروع RAG: تحميل المستندات الخام (Raw Documents).

كل ملف داخل مجلد data/ عبارة عن مستند نصي يحتوي على:
- عنوان الكود (المصدر الذي سيظهر في الاستشهاد/Citation لاحقًا).
- شرح الكود بالعربية.
- الكود الكامل بلغة Python.

هذه المستندات هي "مصادر الحقيقة" (source of truth) التي سيعتمد عليها
المساعد الذكي في الإجابة على أسئلة المستخدم عن أكواد Python الخاصة بالتعلم الآلي.
"""

import os
from dataclasses import dataclass

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@dataclass
class RawDocument:
    source: str        # اسم الملف (يُستخدم كمصدر/Citation)
    title: str          # عنوان الكود المستخرج من أول سطر
    content: str        # محتوى الملف كاملاً


def load_documents(data_dir: str = DATA_DIR) -> list[RawDocument]:
    """يقرأ كل ملفات .txt داخل مجلد البيانات ويعيدها كقائمة مستندات."""
    documents: list[RawDocument] = []

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"مجلد البيانات غير موجود: {data_dir}")

    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(data_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # أول سطر يحتوي عادة على "العنوان: ..."
        first_line = content.strip().splitlines()[0] if content.strip() else filename
        title = first_line.replace("العنوان:", "").strip() if "العنوان:" in first_line else filename

        documents.append(RawDocument(source=filename, title=title, content=content))

    return documents


if __name__ == "__main__":
    docs = load_documents()
    print(f"تم تحميل {len(docs)} مستند(ات) من: {DATA_DIR}\n")
    for d in docs:
        print(f"- [{d.source}] {d.title}  ({len(d.content)} حرف)")
