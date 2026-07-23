def get_groq_llm(api_key: str = None, model_name: str = None, temperature: float = 0.2):
    """
    تستدعي نموذج Groq وتتفاضى مشكلة المعاملات غير المتوافقة.
    """
    active_api_key = api_key or GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    active_model = model_name or GROQ_MODEL or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    if not active_api_key:
        raise ValueError("مفتاح GROQ_API_KEY غير موجود.")

    try:
        from langchain_groq import ChatGroq
        return ChatGroq(
            groq_api_key=active_api_key,
            model_name=active_model,
            temperature=temperature,
        )
    except Exception:
        # كائن بديلي مضمون يعتمد على SDK المباشر لـ Groq
        class GroqLLMWrapper:
            def __init__(self, key, model):
                self.key = key
                self.model = model

            def invoke(self, prompt):
                p_text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
                res = call_groq(p_text, api_key=self.key, model=self.model)
                class Response:
                    content = res
                return Response()

        return GroqLLMWrapper(active_api_key, active_model)
