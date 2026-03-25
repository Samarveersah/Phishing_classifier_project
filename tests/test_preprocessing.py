from src.preprocessing import PAD_TOKEN, UNK_TOKEN, build_char_vocab, encode_url


def test_vocab_contains_special_tokens():
    vocab = build_char_vocab(["https://example.com"])
    assert PAD_TOKEN in vocab
    assert UNK_TOKEN in vocab


def test_encode_url_pads_to_max_length():
    vocab = build_char_vocab(["abc"])
    encoded = encode_url("abc", vocab, max_length=6)
    assert len(encoded) == 6
