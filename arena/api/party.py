"""LAN party helpers: the addresses friends can reach this machine on, and the table waiting for players."""

from __future__ import annotations

import ipaddress
import os
import socket

DEFAULT_WEB_PORT = 5317


def _private(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    return ip.version == 4 and ip.is_private and not ip.is_loopback and not ip.is_link_local


def lan_addresses() -> list[str]:
    """Private IPv4 addresses of this machine, the one on the default route first.

    The route probe sends no packet (UDP connect only picks a source address). It fails when there is no
    route at all, e.g. an offline Wi-Fi network, so host-name lookups back it up.
    """
    found: list[str] = []
    for target in ("10.255.255.255", "192.168.255.255", "8.8.8.8"):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            try:
                probe.connect((target, 9))
                found.append(probe.getsockname()[0])
            except OSError:
                continue
    try:
        found += [info[4][0] for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)]
    except OSError:
        pass
    out: list[str] = []
    for addr in found:
        if _private(addr) and addr not in out:
            out.append(addr)
    return out


def public_urls(env: dict[str, str] | None = None) -> list[str]:
    """Where friends open the table: ``$ARENA_PUBLIC_URL`` if set, else every LAN address on the web port."""
    env = os.environ if env is None else env
    explicit = env.get("ARENA_PUBLIC_URL", "").strip().rstrip("/")
    if explicit:
        return [explicit]
    port = int(env.get("ARENA_WEB_PORT", DEFAULT_WEB_PORT))
    return [f"http://{addr}:{port}" for addr in lan_addresses()]


def same_origin(origin: str, host: str | None) -> bool:
    """A browser page served by this very host (the web server proxies to the API with the Host header intact)."""
    return bool(host) and origin in (f"http://{host}", f"https://{host}")


if __name__ == "__main__":
    print("\n".join(public_urls()))
