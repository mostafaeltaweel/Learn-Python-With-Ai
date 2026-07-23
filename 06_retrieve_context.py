"""
06_retrieve_context.py
------------------------
المرحلة السادسة: استرجاع السياق (Context Retrieval).

بمجرد أن يطرح المستخدم سؤالًا، نقوم بـ:
1. تحويل السؤال إلى متجه بنفس نموذج التضمين المستخدم في بناء المخزن.
2. البحث في مخزن Chroma عن أقرب N قطع (chunks) من حيث التشابه.
3. إعادة القطع المسترجعة مع بيانات المصدر (source/title) الخاصة بها،
   حتى تُستخدم لاحقًا في الاستشهاد بالمصادر (Citations).
"""

from dataclasses import dataclass
from importlib import import_module

_vector_module = import_module("04_vector_representation")
_store_module = import_module("05_create_chroma_store")

embed_query = _vector_module.embed_query
get_chroma_client = _store_module.get_chroma_client
COLLECTION_NAME = _store_module.COLLECTION_NAME

DEFAULT_TOP_K = 4


@dataclass
class RetrievedChunk:
    text: str
    source: str
    title: str
    distance: float


def get_collection():
    client = get_chroma_client()
    return client.get_collection(COLLECTION_NAME)


def retrieve_context(query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
    """يسترجع أقرب top_k قطع نصية متعلقة بسؤال المستخدم."""
    collection = get_collection()
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    retrieved: list[RetrievedChunk] = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for text, meta, dist in zip(documents, metadatas, distances):
        retrieved.append(
            RetrievedChunk(
                text=text,
                source=meta.get("source", "unknown"),
                title=meta.get("title", "unknown"),
                distance=dist,
            )
        )
    return retrieved


def format_context(chunks: list[RetrievedChunk]) -> str:
    """يجهّز نص السياق المُجمّع من القطع المسترجعة، مع ترقيم كل مصدر."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[مصدر {i}: {c.title} | الملف: {c.source}]\n{c.text}")
    return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    question = "ازاي اقدر اعمل تصنيف قطط وكلاب باستخدام Transfer Learning؟"
    results = retrieve_context(question)

    print(f"السؤال: {question}\n")
    print(f"تم استرجاع {len(results)} قطعة:\n")
    for r in results:
        print(f"- المصدر: {r.title} ({r.source}) | المسافة: {r.distance:.4f}")
