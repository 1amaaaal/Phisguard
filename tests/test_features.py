from phishguard.features import (FEATURE_NAMES, extract_features, feature_matrix,
                                 levenshtein, shannon_entropy)


def test_feature_vector_shape_and_order():
    m = feature_matrix(["example.com", "http://1.2.3.4/x"])
    assert m.shape == (2, len(FEATURE_NAMES))


def test_scheme_and_www_do_not_change_features():
    # The dataset-artifact guard: these must be identical.
    a = extract_features("http://example.com/login")
    b = extract_features("https://www.example.com/login")
    assert a == b


def test_brand_impersonation():
    assert extract_features("http://paypal.com.verify-account.xyz/")["brand_impersonation"] == 1
    assert extract_features("https://www.paypal.com/signin")["brand_impersonation"] == 0


def test_typosquat():
    assert extract_features("http://paypa1.com/")["typosquat"] == 1
    assert extract_features("http://micros0ft.com/")["typosquat"] == 1
    assert extract_features("http://microsoft.com/")["typosquat"] == 0


def test_red_flags():
    f = extract_features("http://192.168.0.1:8080/secure//login.php?user=a@b.com")
    assert f["has_ip_host"] == 1 and f["has_port"] == 1
    assert f["has_double_slash_path"] == 1 and f["risky_extension"] == 1
    assert f["has_email_in_url"] == 1
    assert extract_features("http://xn--pple-43d.com")["has_punycode"] == 1
    assert extract_features("http://bit.ly/abc")["is_shortener"] == 1
    assert extract_features("http://foo.000webhostapp.com/")["is_free_host"] == 1


def test_helpers():
    assert levenshtein("kitten", "sitting") == 3
    assert shannon_entropy("aaaa") == 0
    assert shannon_entropy("ab") == 1
