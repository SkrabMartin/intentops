"""Shared local-only API address validation; never follow provider URLs or proxies."""

import ipaddress

import httpx


def local_api_url(value: str) -> str:
    url = httpx.URL(value)
    try:
        local = ipaddress.ip_address(url.host).is_loopback
    except ValueError:
        local = url.host in ("localhost", "api")
    if (
        url.scheme != "http"
        or not local
        or url.userinfo
        or url.query
        or url.fragment
        or url.path not in ("", "/")
    ):
        raise ValueError("Only a local HTTP API origin is allowed")
    return str(url).rstrip("/")
