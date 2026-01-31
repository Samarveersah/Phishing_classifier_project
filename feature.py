import re
from urllib.parse import urlparse


def extract_features(url):
    features = []

    # Length of URL
    features.append(len(url))

    # Count special characters
    features.append(url.count('.'))
    features.append(url.count('-'))
    features.append(url.count('@'))
    features.append(url.count('?'))
    features.append(url.count('%'))

    # Check IP address in URL
    ip_pattern = re.compile(r'\d+\.\d+\.\d+\.\d+')
    features.append(1 if ip_pattern.search(url) else 0)

    # URL structure info
    parsed = urlparse(url)
    features.append(len(parsed.netloc))
    features.append(len(parsed.path))

    return features
