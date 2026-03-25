from src.feature_extraction import extract_url_features


def test_extract_url_features_shape():
    features = extract_url_features("https://example.com/login?user=test")
    assert features.shape == (16,)


def test_extract_url_features_detects_ip():
    features = extract_url_features("http://192.168.0.1/reset-password")
    assert features[9] == 1
