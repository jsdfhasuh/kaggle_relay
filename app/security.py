import re


TOKEN_PATTERNS = [
    re.compile(r"(Authorization:\s*Bearer\s+)[^\s]+", re.IGNORECASE),
    re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+", re.IGNORECASE),
    re.compile(r"KGAT_[A-Za-z0-9_\-]+"),
    re.compile(r"(?i)(KAGGLE_(?:API_TOKEN|KEY|USERNAME)=)[^\s]+"),
]


def redact_secrets(text: str) -> str:
    value = str(text or "")
    for pattern in TOKEN_PATTERNS:
        value = pattern.sub(lambda match: f"{match.group(1)}***" if match.groups() else "***", value)
    return value

