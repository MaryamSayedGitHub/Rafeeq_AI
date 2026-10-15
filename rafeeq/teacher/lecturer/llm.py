import os
import sys

from rafeeq.teacher.lecturer.settings import LLM_TEMPERATURE, OLLAMA_BASE_URL

PROVIDERS = {
    "groq":   {"key": "GROQ_API_KEY",   "model_env": "GROQ_MODEL"},
    "gemini": {"key": "GOOGLE_API_KEY", "model_env": "GEMINI_MODEL"},
    "xai":    {"key": "XAI_API_KEY",    "model_env": "XAI_MODEL"},
    "ollama": {"key": None,             "model_env": "OLLAMA_MODEL"},
}

_chosen_provider: str | None = None


def _missing(provider: str) -> list[str]:
    """Env vars that are still empty for this provider."""
    cfg = PROVIDERS[provider]
    needed = [cfg["model_env"]] + ([cfg["key"]] if cfg["key"] else [])
    return [name for name in needed if not os.getenv(name)]


def _validate(provider: str) -> str:
    provider = provider.strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(PROVIDERS)}")
    missing = _missing(provider)
    if missing:
        raise RuntimeError(f"Missing in .env for provider '{provider}': {', '.join(missing)}")
    return provider


def _ask(ready: list[str]) -> str:
    print("Which LLM do you want to use?")
    for i, p in enumerate(ready, 1):
        print(f"  {i}) {p}")
    while True:
        choice = input("Number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(ready):
            return ready[int(choice) - 1]
        print(f"Please enter a number between 1 and {len(ready)}.")


def choose_provider(ask: bool = False) -> str:
    """
    Priority: LLM_PROVIDER from .env -> ask in terminal (only if ask=True) -> first ready provider.
    The choice is cached, so it happens once per run.
    """
    global _chosen_provider
    if _chosen_provider:
        return _chosen_provider

    provider = os.getenv("LLM_PROVIDER", "").strip().lower()

    if not provider:
        ready = [p for p in PROVIDERS if not _missing(p)]
        if not ready:
            raise RuntimeError(
                "No LLM provider is configured. Set a provider's API key and model name in .env."
            )
        if ask and sys.stdin.isatty() and len(ready) > 1:
            provider = _ask(ready)
        else:
            provider = ready[0]
            print(f"[llm] LLM_PROVIDER is not set, using '{provider}'.", file=sys.stderr)

    _chosen_provider = _validate(provider)
    return _chosen_provider


def get_llm(provider: str | None = None, temperature: float | None = None):
    """
    get_llm()                              -> the globally chosen provider
    get_llm(provider="gemini")             -> force a provider for this agent
    get_llm(temperature=0.7)               -> override the temperature
    """
    provider = _validate(provider) if provider else choose_provider()
    temperature = LLM_TEMPERATURE if temperature is None else temperature
    model = os.environ[PROVIDERS[provider]["model_env"]]

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=temperature)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    if provider == "xai":
        from langchain_xai import ChatXAI
        return ChatXAI(model=model, temperature=temperature)

    from langchain_ollama import ChatOllama
    return ChatOllama(model=model, base_url=OLLAMA_BASE_URL, temperature=temperature)