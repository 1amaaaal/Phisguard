"""URL parsing helpers shared by feature extraction, data prep and the API.

Everything here is pure string work: no DNS, WHOIS or HTTP requests, so a
prediction is instant, works offline, and never touches the suspicious site.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

# Two-label public suffixes common enough to matter. A full Public Suffix List
# (e.g. the `tldextract` package) is more exact; this keeps the project
# dependency-free and offline.
MULTI_PART_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "ltd.uk", "plc.uk", "me.uk",
    "com.au", "net.au", "org.au", "edu.au", "gov.au",
    "co.nz", "org.nz", "co.za", "co.in", "net.in", "org.in", "gov.in", "ac.in",
    "co.jp", "ne.jp", "or.jp", "ac.jp", "co.kr", "or.kr",
    "com.br", "net.br", "org.br", "gov.br", "com.mx", "com.ar", "com.co",
    "com.cn", "net.cn", "org.cn", "com.hk", "com.tw", "com.sg", "com.my",
    "com.tr", "com.ua", "net.ua", "org.ua", "gov.ua", "in.ua", "kiev.ua",
    "com.pl", "com.ru", "com.vn", "com.ng", "com.pk",
    "com.eg", "com.sa", "com.pe", "com.ph", "co.id", "or.id", "web.id",
    "co.il", "co.th", "in.th",
}

_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


@dataclass(frozen=True)
class ParsedURL:
    raw: str
    host: str               # lower-case hostname without port or "www."
    port: int | None
    path: str
    query: str
    fragment: str
    registered_domain: str  # e.g. "paypal.com" for "www.login.paypal.com"
    subdomain: str          # e.g. "login" for "www.login.paypal.com"
    tld: str                # e.g. "com", "co.uk"
    is_ip: bool


def ensure_scheme(url: str) -> str:
    url = url.strip()
    return url if _SCHEME_RE.match(url) else "http://" + url


def normalize_url(url: str) -> str:
    """Canonical form fed to the model: no scheme, no leading "www.".

    Removing these is deliberate. In the training data almost every benign URL
    is "http://" with no "www.", while phishing URLs are a mix. Left in, the
    model would learn "https means phishing", which is a dataset artifact and
    not something that holds up in the real world.
    """
    url = _SCHEME_RE.sub("", url.strip())
    if url.lower().startswith("www."):
        url = url[4:]
    return url.rstrip("/") if url.count("/") <= 1 else url


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        pass
    # Hex/decimal-encoded IPs such as 0x7f000001 or 3232235777.
    return bool(re.fullmatch(r"(0x[0-9a-f]+|\d{8,10})", host))


def split_host(host: str) -> tuple[str, str, str]:
    """Return (subdomain, registered_domain, tld) for a hostname."""
    labels = [label for label in host.split(".") if label]
    if len(labels) < 2:
        return "", host, ""
    last_two = ".".join(labels[-2:])
    suffix_len = 2 if last_two in MULTI_PART_SUFFIXES and len(labels) >= 3 else 1
    tld = ".".join(labels[-suffix_len:])
    registered = ".".join(labels[-(suffix_len + 1):])
    subdomain = ".".join(labels[:-(suffix_len + 1)])
    return subdomain, registered, tld


def parse_url(url: str) -> ParsedURL:
    raw = url.strip()
    try:
        parts = urlsplit(ensure_scheme(raw))
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:  # malformed netloc, e.g. a bad port or bracket
        parts = urlsplit(ensure_scheme(raw).replace("[", "").replace("]", ""))
        host = parts.netloc.rsplit("@", 1)[-1].split(":")[0].lower()
        port = None
    if host.startswith("www."):
        host = host[4:]
    is_ip = _is_ip(host)
    if is_ip:
        subdomain, registered, tld = "", host, ""
    else:
        subdomain, registered, tld = split_host(host)
    return ParsedURL(
        raw=raw, host=host, port=port, path=parts.path, query=parts.query,
        fragment=parts.fragment, registered_domain=registered,
        subdomain=subdomain, tld=tld, is_ip=is_ip,
    )


def registered_domain(url: str) -> str:
    return parse_url(url).registered_domain
