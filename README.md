# مساعد شرح أكواد Python بالعربية (مشروع RAG)

مشروع **Retrieval-Augmented Generation (RAG)** بسيط ومبني بالكامل بملفات Python
(بدون Notebooks)، يتّبع تسلسل المعمل الكامل:

```
documents -> preprocessing -> chunking -> vector representation ->
vector store -> context retrieval -> prompting -> Streamlit UI
```

المشروع يجيب على أسئلة المستخدم **بالعربية** عن مجموعة أكواد Python الخاصة
بالتعلم الآلي والتعلم العميق، بالاعتماد فقط على مصادر البيانات المُرفقة،
مع ذكر المصدر (Citation) في نهاية كل إجابة.

---

## 1) مصدر البيانات (Dataset)

البيانات عبارة عن 8 ملفات نصية داخل مجلد `data/`، كل ملف يحتوي على:
عنوان الكود + شرح باللغة العربية لما يفعله الكود خطوة بخطوة + الكود الكامل.

| # | عنوان الكود (Title) | اسم ملف المصدر (source) | الموضوع الرئيسي |
|---|----------------------|---------------------------|-------------------|
| 1 | كود تدريب نموذج YOLO على بيانات Oxford Pet Dataset | `doc_01_yolo_oxford_pets.txt` | كشف الكائنات (Object Detection) |
| 2 | كود التصنيف والتحديد المبسط للكائنات باستخدام VGG16 | `doc_02_vgg16_classification_bbox.txt` | تصنيف + Bounding Box بنموذج متعدد المهام |
| 3 | كود الشبكة العصبية الأساسية (فيديو المقدمة) | `doc_03_basic_neural_network.txt` | خلية عصبية واحدة / Logistic Regression |
| 4 | كود الـ Autoencoder لإزالة التشويش من الصور | `doc_04_autoencoder_denoising.txt` | Denoising Autoencoder على MNIST |
| 5 | كود الشبكة العصبية المتكررة (RNN) للتنبؤ بسلسلة زمنية | `doc_05_rnn_time_series.txt` | SimpleRNN وتحليل سلاسل زمنية |
| 6 | كود الشبكة العصبية التلافيفية (CNN) للتعرف على أرقام مبسطة | `doc_06_cnn_digits.txt` | CNN مبسّطة على بيانات 5x5 |
| 7 | كود التعلم بالنقل باستخدام MobileNet لتصنيف القطط والكلاب | `doc_07_mobilenet_transfer_learning.txt` | Transfer Learning للتصنيف الثنائي |
| 8 | كود U-Net لتجزئة صور الأورام | `doc_08_unet_segmentation.txt` | Semantic Segmentation طبي |

> يمكن إضافة أي عدد آخر من ملفات `.txt` داخل `data/` وسيتم التقاطها تلقائيًا
> في المرحلة الأولى (`01_documents.py`) دون تعديل أي كود آخر.

---

## 2) هيكل المشروع

```
rag_project/
├── data/                          # المستندات الخام (8 ملفات نصية عربية)
├── 01_documents.py                # تحميل المستندات الخام
├── 02_preprocessing.py            # تنظيف النصوص
├── 03_chunking.py                 # تقطيع النصوص إلى Chunks
├── 04_vector_representation.py    # تمثيل متجهي (Embeddings) محلي
├── 05_create_chroma_store.py      # بناء مخزن Chroma الدائم
├── 06_retrieve_context.py         # استرجاع السياق الأقرب للسؤال
├── 07_prompting.py                # بناء الـ Prompt + استدعاء Groq API
├── streamlit_app.py               # واجهة المستخدم النهائية
├── requirements.txt
├── .env.example                   # مثال لملف البيئة (بدون مفتاح حقيقي)
├── .gitignore
└── README.md
```

---

## 3) خط أنابيب المعالجة (Pipeline)

| المرحلة | الملف | الوظيفة |
|---|---|---|
| 1. Documents | `01_documents.py` | قراءة ملفات `data/*.txt` كمستندات خام |
| 2. Preprocessing | `02_preprocessing.py` | تنظيف النص (مسافات، أسطر فارغة) |
| 3. Chunking | `03_chunking.py` | تقطيع كل مستند إلى قطع ~900 حرف بتداخل 150 حرف |
| 4. Vector Representation | `04_vector_representation.py` | تمثيل كل قطعة بمتجه عبر `sentence-transformers` (نموذج متعدد اللغات مجاني) |
| 5. Vector Store | `05_create_chroma_store.py` | تخزين المتجهات في قاعدة `ChromaDB` دائمة على القرص (`chroma_db/`) |
| 6. Context Retrieval | `06_retrieve_context.py` | استرجاع أقرب `top_k` قطع لسؤال المستخدم |
| 7. Prompting | `07_prompting.py` | بناء Prompt عربي + استدعاء نموذج **Groq** لتوليد الإجابة مع المصادر |
| 8. UI | `streamlit_app.py` | واجهة محادثة تعرض الإجابة + القطع المسترجعة (Citations) |

يمكن تشغيل كل ملف بشكل مستقل للاختبار:

```bash
python 01_documents.py
python 02_preprocessing.py
python 03_chunking.py
python 04_vector_representation.py
python 05_create_chroma_store.py   # يبني قاعدة Chroma (خطوة مطلوبة قبل التشغيل)
python 06_retrieve_context.py
python 07_prompting.py             # يحتاج GROQ_API_KEY في .env
```

---

## 4) التشغيل المحلي (Local Setup)

```bash
# 1) إنشاء بيئة افتراضية
python -m venv venv
source venv/bin/activate      # على ويندوز: venv\Scripts\activate

# 2) تثبيت المتطلبات
pip install -r requirements.txt

# 3) نسخ ملف البيئة وتعبئة المفتاح الحقيقي
cp .env.example .env
# افتح .env وضع مفتاح Groq API الحقيقي الخاص بك في GROQ_API_KEY

# 4) بناء مخزن المتجهات (مرة واحدة، أو بعد أي تعديل على data/)
python 05_create_chroma_store.py

# 5) تشغيل التطبيق
streamlit run streamlit_app.py
```

الحصول على مفتاح Groq API مجانًا من: https://console.groq.com/keys

---

## 5) قواعد أمان مفتاح الـ API

- **لا يُكتب المفتاح الحقيقي داخل أي ملف Python.** كل الملفات تقرأ المفتاح
  من متغير بيئة `GROQ_API_KEY` فقط (عبر `.env` محليًا أو `st.secrets` عند النشر).
- **لا يُرفع ملف `.env` الحقيقي** إلى GitHub — هو مُستثنى في `.gitignore`،
  ويوجد بدلاً منه `.env.example` بدون مفتاح حقيقي.
- عند النشر على Streamlit Cloud، يُضاف المفتاح من خلال **Secrets** فقط
  (TOML)، وليس داخل الكود.

---

## 6) النشر على Streamlit Cloud

1. ارفع المشروع (بدون `.env` وبدون `chroma_db/`) إلى مستودع GitHub.
2. من [share.streamlit.io](https://share.streamlit.io) أنشئ تطبيقًا جديدًا
   واختر المستودع والملف الرئيسي `streamlit_app.py`.
3. من داخل التطبيق: **Manage app → Settings → Secrets**، وأضف:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
GROQ_MODEL = "llama-3.1-8b-instant"
```

4. تأكد أن `streamlit_app.py` يقرأ المفتاح من `st.secrets` عند عدم توفره
   في البيئة (هذا مُطبّق بالفعل في الكود):

```python
try:
    if not rag.GROQ_API_KEY:
        rag.GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
    rag.GROQ_MODEL = st.secrets.get("GROQ_MODEL", rag.GROQ_MODEL)
except Exception:
    pass
```

5. ملاحظة مهمة: مخزن `chroma_db/` غير مرفوع على GitHub، لذا يجب إما:
   - تشغيل `python 05_create_chroma_store.py` تلقائيًا عند إقلاع التطبيق (يمكن
     استدعاؤه من `streamlit_app.py` عند عدم وجود المجلد)، أو
   - رفعه يدويًا ضمن المستودع إن كان صغير الحجم (كما هو الحال هنا).

---

## 7) قائمة التحقق النهائية (Final Checklist)

- [x] جميع ملفات Python المطلوبة موجودة (01 → 07 + `streamlit_app.py`).
- [x] `requirements.txt` موجود.
- [x] لا يوجد مفتاح API حقيقي داخل الملفات أو المستودع.
- [x] إعدادات Streamlit Secrets بصيغة TOML صحيحة.
- [x] التطبيق يعمل ويعرض إجابة مبنية على السياق المسترجع.
- [x] الإجابة تستخدم السياق المسترجع (Context) فقط.
- [x] الإجابة تذكر المصادر (Citations) في النهاية.

---

## 8) التسليم النهائي (Submission)

يجب تسليم:
1. ملف **ZIP** لكامل المشروع.
2. رابط **مستودع GitHub**.
3. رابط **تطبيق Streamlit** المنشور.
