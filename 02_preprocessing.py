"""
02_preprocessing.py
--------------------
المرحلة الثانية: تنظيف النصوص (Preprocessing) قبل التقطيع (Chunking).

نقوم هنا بـ:
- إزالة المسافات والأسطر الفارغة الزائدة.
- توحيد الفواصل بين الأسطر.
- الإبقاء على الكود البرمجي كما هو (لا نزيل الرموز الخاصة بلغة Python
  حتى لا نُتلف صلاحية الكود عند عرضه للمستخدم).
"""

import re

from importlib import import_module

_documents_module = import_module("01_documents")
load_documents = _documents_module.load_documents
RawDocument = _documents_module.RawDocument


def clean_text(text: str) -> str:
    """تنظيف بسيط للنص: إزالة المسافات الزائدة وتوحيد الأسطر الفارغة."""
    text = text.replace("\r\n", "\n")
    # إزالة أكثر من سطرين فارغين متتاليين
    text = re.sub(r"\n{3,}", "\n\n", text)
    # إزالة المسافات في نهاية كل سطر
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def preprocess_documents(documents: list[RawDocument]) -> list[RawDocument]:
    """يطبق التنظيف على كل مستند ويعيد نسخة جديدة نظيفة."""
    cleaned = []
    for doc in documents:
        cleaned.append(
            RawDocument(
                source=doc.source,
                title=doc.title,
                content=clean_text(doc.content),
            )
        )
    return cleaned


if __name__ == "__main__":
    raw_docs = load_documents()
    clean_docs = preprocess_documents(raw_docs)

    print(f"تمت معالجة {len(clean_docs)} مستند(ات).\n")
    for d in clean_docs:
        print(f"- [{d.source}] {d.title}  ({len(d.content)} حرف بعد التنظيف)")
