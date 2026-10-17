from phishguard.urltools import normalize_url, parse_url


def test_normalize_strips_scheme_and_www():
    assert normalize_url("https://www.example.com/a/b") == "example.com/a/b"
    assert normalize_url("http://example.com/") == "example.com"
    assert normalize_url("example.com") == "example.com"


def test_registered_domain_and_subdomain():
    p = parse_url("https://login.secure.paypal.com.evil.xyz/path?q=1")
    assert p.registered_domain == "evil.xyz"
    assert p.subdomain == "login.secure.paypal.com"
    assert p.tld == "xyz"


def test_multi_part_suffix():
    p = parse_url("http://shop.bbc.co.uk/")
    assert p.registered_domain == "bbc.co.uk"
    assert p.tld == "co.uk"


def test_ip_hosts():
    assert parse_url("http://192.168.10.5/login").is_ip
    assert parse_url("http://[2001:db8::1]/").is_ip
    assert parse_url("http://3232235777/").is_ip
    assert not parse_url("http://example.com").is_ip


def test_malformed_url_does_not_crash():
    for url in ["http://exa mple.com:99999/", "http://[bad/", "::::", "a"]:
        parse_url(url)
