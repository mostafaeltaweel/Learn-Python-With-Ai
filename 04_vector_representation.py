"""
04_vector_representation.py
-----------------------------
المرحلة الرابعة: تحويل كل قطعة نصية (Chunk) إلى تمثيل متجهي (Embedding).

نستخدم نموذج تضمين محلي مجاني من مكتبة sentence-transformers
(multilingual لأن بياناتنا عربية وإنجليزية معًا) بدلاً من استخدام API خارجي،
لتجنّب أي تكلفة إضافية أو الحاجة لمفتاح API في هذه المرحلة.

النموذج المستخدم:
    paraphrase-multilingual-MiniLM-L12-v2
وهو نموذج صغير وسريع ويدعم اللغة العربية والإنجليزية معًا.
"""

from functools import lru_cache

from importlib import import_module

_chunking_module = import_module("03_chunking")
build_chunks = _chunking_module.build_chunks

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def get_embedding_model():
    """يحمّل نموذج التضمين مرة واحدة فقط (Singleton) لتوفير الوقت والذاكرة."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """يحوّل قائمة نصوص إلى قائمة متجهات (vectors)."""
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """يحوّل سؤال المستخدم إلى متجه واحد بنفس النموذج المستخدم للمستندات."""
    return embed_texts([query])[0]


if __name__ == "__main__":
    chunks = build_chunks()
    texts = [c.text for c in chunks]
    vectors = embed_texts(texts)

    print(f"تم تمثيل {len(vectors)} قطعة كمتجهات.")
    print(f"أبعاد كل متجه (Vector Dimension): {len(vectors[0])}")
