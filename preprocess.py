import re

import emoji


_URL_PATTERN = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_MENTION_PATTERN = re.compile(r"(?<![\w@.])@[A-Za-z0-9_]+\b")


def preprocess_text(text: str) -> str:
    """Normalize text for the MBG demo classifier."""
    value = "" if text is None else str(text)
    value = value.lower()
    value = _MENTION_PATTERN.sub("@USER", value)
    value = _URL_PATTERN.sub("HTTPURL", value)
    value = emoji.demojize(value, language="en")
    return " ".join(value.split()).strip()


def preprocess_pair(text: str, parent_text: str = None) -> dict:
    """Normalize a post and its optional parent post."""
    return {
        "text": preprocess_text(text),
        "parent_text": preprocess_text(parent_text),
    }
