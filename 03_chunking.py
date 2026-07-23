"""
03_chunking.py
---------------
المرحلة الثالثة: تقطيع المستندات إلى قطع أصغر (Chunks).

نستخدم أسلوب "نافذة منزلقة" مبنية على عدد الأحرف مع تداخل (overlap) بسيط،
حتى لا تُقطع فكرة الشرح أو سطر الكود في منتصفه بدون سياق.
كل Chunk يحتفظ بمصدره الأصلي (source) وعنوانه (title) حتى نستطيع
الاستشهاد بالمصدر (Citation) لاحقًا عند توليد الإجابة.
"""

from dataclasses import dataclass
from importlib import import_module

_documents_module = import_module("01_documents")
_preprocessing_module = import_module("02_preprocessing")

load_documents = _documents_module.load_documents
preprocess_documents = _preprocessing_module.preprocess_documents

CHUNK_SIZE = 900      # عدد الأحرف التقريبي لكل قطعة
CHUNK_OVERLAP = 150   # عدد أحرف التداخل بين القطع المتتالية


@dataclass
class Chunk:
    chunk_id: str
    source: str
    title: str
    text: str


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """يقطع نصًا طويلًا إلى قطع بحجم chunk_size حرف مع تداخل overlap حرف."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def build_chunks(documents=None) -> list[Chunk]:
    """يحول قائمة المستندات إلى قائمة Chunks جاهزة للتمثيل المتجهي."""
    if documents is None:
        documents = preprocess_documents(load_documents())

    all_chunks: list[Chunk] = []
    for doc in documents:
        pieces = chunk_text(doc.content)
        for i, piece in enumerate(pieces):
            all_chunks.append(
                Chunk(
                    chunk_id=f"{doc.source}::chunk_{i}",
                    source=doc.source,
                    title=doc.title,
                    text=piece,
                )
            )
    return all_chunks


if __name__ == "__main__":
    chunks = build_chunks()
    print(f"تم إنشاء {len(chunks)} قطعة (chunk) من المستندات.\n")
    for c in chunks[:5]:
        print(f"- {c.chunk_id}  |  العنوان: {c.title}  |  طول النص: {len(c.text)}")
