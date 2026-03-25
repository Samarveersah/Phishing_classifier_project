import re
from urllib.parse import urlparse

import numpy as np


IP_PATTERN = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
SHORTENER_PATTERN = re.compile(
    r"(bit\.ly|tinyurl\.com|goo\.gl|t\.co|ow\.ly|is\.gd|buff\.ly)",
    re.IGNORECASE,
)


FEATURE_NAMES = [
    "url_length",
    "dot_count",
    "hyphen_count",
    "at_count",
    "question_count",
    "percent_count",
    "slash_count",
    "digit_ratio",
    "has_https",
    "has_ip_address",
    "subdomain_count",
    "domain_length",
    "path_length",
    "query_length",
    "fragment_length",
    "has_shortener",
]


def extract_url_features(url: str) -> np.ndarray:
    url = (url or "").strip()
    parsed = urlparse(url if "://" in url else f"http://{url}")
    hostname = parsed.netloc.split(":")[0]

    digit_ratio = sum(character.isdigit() for character in url) / max(len(url), 1)
    has_https = 1 if parsed.scheme.lower() == "https" else 0
    has_ip_address = 1 if IP_PATTERN.match(hostname) else 0
    subdomain_count = max(hostname.count(".") - 1, 0) if hostname else 0
    has_shortener = 1 if SHORTENER_PATTERN.search(hostname) else 0

    features = np.array(
        [
            len(url),
            url.count("."),
            url.count("-"),
            url.count("@"),
            url.count("?"),
            url.count("%"),
            url.count("/"),
            digit_ratio,
            has_https,
            has_ip_address,
            subdomain_count,
            len(hostname),
            len(parsed.path),
            len(parsed.query),
            len(parsed.fragment),
            has_shortener,
        ],
        dtype=np.float32,
    )
    return features
