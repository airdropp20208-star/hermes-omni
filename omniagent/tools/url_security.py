"""Shared URL safety checks for network-capable tools."""

import ipaddress
import socket
from urllib.parse import urljoin, urlparse


class URLSecurityError(ValueError):
    """Raised when a URL is not safe for tool-side HTTP access."""


def _format_ip(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str:
    return str(ip_obj)


def _is_global_ip(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return ip_obj.is_global


def _resolved_ip_addresses(
    hostname: str,
    port: int | None,
) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        return [ipaddress.ip_address(hostname)]
    except ValueError:
        pass

    try:
        addr_infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise URLSecurityError(f"Could not resolve hostname: {hostname}") from exc

    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    seen: set[str] = set()
    for info in addr_infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        ip_text = sockaddr[0]
        try:
            ip_obj = ipaddress.ip_address(ip_text)
        except ValueError:
            continue
        if ip_text not in seen:
            seen.add(ip_text)
            addresses.append(ip_obj)

    if not addresses:
        raise URLSecurityError(f"Could not resolve hostname: {hostname}")
    return addresses


def validate_public_http_url(url: str) -> None:
    """Reject URLs that do not target a public http(s) address."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise URLSecurityError("URL must start with http:// or https://")
    if not parsed.hostname:
        raise URLSecurityError(f"Invalid URL: {url}")

    port = parsed.port
    for ip_obj in _resolved_ip_addresses(parsed.hostname, port):
        if not _is_global_ip(ip_obj):
            raise URLSecurityError(
                "URL points to a private/internal network "
                f"(SSRF protection): {parsed.hostname} -> {_format_ip(ip_obj)}"
            )


def validate_redirect_url(source_url: str, location: str) -> str:
    """Resolve and validate a redirect target."""
    if not location:
        raise URLSecurityError("Redirect target is empty")
    redirected_url = urljoin(source_url, location)
    validate_public_http_url(redirected_url)
    return redirected_url
