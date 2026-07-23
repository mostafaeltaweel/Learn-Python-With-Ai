"""
05_create_chroma_store.py
---------------------------
المرحلة الخامسة: بناء مخزن المتجهات (Vector Store) باستخدام ChromaDB.

نأخذ كل الـ Chunks، نحوّلها إلى متجهات، ثم نخزّنها في قاعدة بيانات Chroma
محلية دائمة (Persistent) على القرص، حتى لا نُعيد بناء التمثيل المتجهي
في كل مرة يعمل فيها التطبيق.
"""

import os
from importlib import import_module

_chunking_module = import_module("03_chunking")
_vector_module = import_module("04_vector_representation")

build_chunks = _chunking_module.build_chunks
embed_texts = _vector_module.embed_texts

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR = os.path.join(PROJECT_DIR, "chroma_db")
COLLECTION_NAME = "python_code_explanations_ar"


def get_chroma_client():
    """ينشئ أو يجلب العميل الخاص بـ ChromaDB على القرص."""
    import chromadb
    return chromadb.PersistentClient(path=CHROMA_DB_DIR)


def create_or_reset_collection(client):
    """ينشئ Collection جديدة، ويحذف القديمة إن وُجدت لإعادة البناء من الصفر."""
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def build_vector_store():
    """يبني مخزن المتجهات الكامل من مستندات مجلد data/ وحتى Chroma."""
    chunks = build_chunks()
    texts = [c.text for c in chunks]
    ids = [c.chunk_id for c in chunks]
    metadatas = [{"source": c.source, "title": c.title} for c in chunks]

    print(f"جاري تمثيل {len(texts)} قطعة كمتجهات...")
    embeddings = embed_texts(texts)

    client = get_chroma_client()
    collection = create_or_reset_collection(client)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"تم حفظ {collection.count()} قطعة في مخزن Chroma بالمسار: {CHROMA_DB_DIR}")
    return collection


def create_or_load_vector_store(chunks=None, embedding_model=None, persist_directory=None):
    """
    الدالة المطلوبة لربط streamlit_app.py:
    تحمّل قاعدة البيانات إن كانت موجودة مسبقاً وبها بيانات، أو تقوم ببنائها لأول مرة.
    """
    client = get_chroma_client()
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        if collection.count() > 0:
            print(f"✅ تم تحميل قاعدة ChromaDB الموجودة بنجاح ({collection.count()} قطعة).")
            return collection
    except Exception:
        pass

    print("🌐 لم يتم العثور على قاعدة بيانات جهزة، جاري البناء الآن...")
    return build_vector_store()


if __name__ == "__main__":
    build_vector_store()
