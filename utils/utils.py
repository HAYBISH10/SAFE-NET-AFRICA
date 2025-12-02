import os
import re
from datetime import datetime
from typing import Tuple


LOG_DIR = os.path.join("logs")
LOG_FILE = os.path.join(LOG_DIR, "error.log")


def ensure_log_dir() -> None:
    """Ensure the logs directory exists."""
    os.makedirs(LOG_DIR, exist_ok=True)


def log_error(message: str) -> None:
    """
    Log internal errors for debugging.
    IMPORTANT: Do NOT log user message content.
    """
    ensure_log_dir()
    timestamp = datetime.utcnow().isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def clean_text(text: str) -> str:
    """
    Basic, safe text cleaning.
    NOTE: We keep this intentionally light so we don't remove important info.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def clamp_probability(prob: float) -> float:
    """
    Ensure probability is between 0 and 1, just in case.
    """
    try:
        prob = float(prob)
    except Exception:
        prob = 0.0
    return max(0.0, min(1.0, prob))


def risk_level(prob: float) -> str:
    """
    Convert a probability (0–1) into a simple risk level.
    """
    prob = clamp_probability(prob)
    if prob >= 0.75:
        return "high"
    elif prob >= 0.5:
        return "medium"
    else:
        return "low"


# ---------- PHISHING / SCAM EXPLANATIONS ----------

def explain_phishing(prob: float) -> str:
    """
    Human-friendly explanation for phishing probability.
    """
    prob = clamp_probability(prob)
    level = risk_level(prob)

    if level == "high":
        return (
            "⚠️ The model sees several patterns that are common in spam or phishing, "
            "such as suspicious wording, structure, or style. Treat this message as risky."
        )
    elif level == "medium":
        return (
            "⚠️ The message has some features that look similar to spam or phishing. "
            "It might be safe, but you should double-check before trusting it."
        )
    else:
        return (
            "✅ The message does not strongly match common spam or phishing patterns. "
            "However, always stay cautious with links and personal information."
        )


def guidance_phishing(prob: float) -> str:
    """
    Short safety guidance for phishing risk.
    """
    prob = clamp_probability(prob)
    level = risk_level(prob)

    if level == "high":
        return (
            "- Do **not** click any links.\n"
            "- Do **not** share passwords, OTPs, or personal details.\n"
            "- Verify the sender using official channels.\n"
            "- Consider blocking or reporting the sender."
        )
    elif level == "medium":
        return (
            "- Be cautious and verify the message from another source.\n"
            "- Check spelling, sender address, and unusual requests.\n"
            "- Avoid sharing sensitive information until you are sure."
        )
    else:
        return (
            "- It looks mostly safe, but still:\n"
            "  - Avoid sharing sensitive details casually.\n"
            "  - Be careful with links and attachments.\n"
            "  - Trust your instincts if something feels off."
        )


# ---------- TOXICITY / ABUSE EXPLANATIONS ----------

def explain_toxic(prob: float) -> str:
    """
    Human-friendly explanation for toxicity probability.
    """
    prob = clamp_probability(prob)
    level = risk_level(prob)

    if level == "high":
        return (
            "⚠️ This message is very likely to be abusive or harmful. "
            "It contains strong signals of toxic or aggressive language."
        )
    elif level == "medium":
        return (
            "⚠️ This message shows some signs of abusive or harmful language. "
            "It may not be fully severe, but it could still feel hurtful or unsafe."
        )
    else:
        return (
            "✅ The message does not strongly match patterns of abusive language "
            "based on the model. However, your feelings are still valid."
        )


def guidance_toxic(prob: float) -> str:
    """
    Survivor-safe guidance for handling abusive content.
    """
    prob = clamp_probability(prob)
    level = risk_level(prob)

    base = (
        "- Your feelings matter. If this message hurt you, that is valid.\n"
        "- You deserve to feel safe online.\n"
    )

    if level == "high":
        extra = (
            "- Consider **blocking or muting** the sender.\n"
            "- **Keep evidence** (screenshots, dates, usernames) if you may report later.\n"
            "- Reach out to **trusted friends, family, or support organisations**.\n"
        )
    elif level == "medium":
        extra = (
            "- You may want to reduce contact with the sender.\n"
            "- Save the message if you feel you might need it.\n"
            "- Talk to someone you trust about how this made you feel.\n"
        )
    else:
        extra = (
            "- Even if the model does not flag it as strongly abusive, "
            "you can still set boundaries.\n"
            "- If this is part of a repeated pattern, consider talking to someone you trust.\n"
        )

    disclaimer = (
        "\n> This tool does **not** replace professional, legal, or psychological support. "
        "If you are in immediate danger, please seek local emergency help where possible."
    )

    return base + extra + disclaimer


# ---------- PREDICTION HELPERS ----------

def predict_with_model(
    text: str,
    model,
) -> Tuple[float, str]:
    """
    Generic prediction helper.

    Returns:
        (probability_of_positive_class, error_message)
        error_message is empty string "" if everything is OK.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return 0.0, "Empty text after cleaning."

    try:
        # Assumes a scikit-learn Pipeline with predict_proba
        proba = model.predict_proba([cleaned])[0][1]
        proba = clamp_probability(proba)
        return proba, ""
    except Exception as e:
        log_error(f"Model prediction failed: {e}")
        return 0.0, "Model error. Please try again later."
