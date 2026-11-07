"""Who may use the studio's local servers (the pages on http://127.0.0.1:8000, the app window's own page server,
camera.mjpg and the WebSocket on ws://127.0.0.1:8765).

They listen on 127.0.0.1 only, but any website open in a browser on this PC can still reach 127.0.0.1: it could
open the WebSocket (read the tracking data, send settings and key presses), and with a DNS-rebinding domain even
read camera.mjpg as "its own" address. So:
  - a browser request has to come from one of Avatar Studio's own pages (an Origin of http://127.0.0.1:<port>,
    http://localhost:<port> or http://[::1]:<port>); programs that aren't browsers send no Origin and are fine,
    since they already run on this PC;
  - the Host header has to be a local address (a rebinding domain like evil.example:8000 is refused);
  - pages can't be framed by other sites.
"""
import re

_LOCAL_HOST = re.compile(r"^(127\.0\.0\.1|localhost|\[::1\])(:\d{1,5})?$", re.I)


def local_host(host):
    """A Host header (or an origin's host part) that names this PC: 127.0.0.1, localhost or [::1], any port."""
    return bool(host) and bool(_LOCAL_HOST.match(str(host).strip()))


def local_origin(origin):
    """An Origin header that's one of Avatar Studio's own pages, or none at all (not a browser)."""
    if origin is None:
        return True
    m = re.match(r"^http://([^/]+)$", str(origin).strip(), re.I)
    return bool(m) and local_host(m.group(1))


def allowed(headers):
    """Whether an HTTP / WebSocket request with these headers (anything with .get) may be answered."""
    if not local_host(headers.get("Host")):
        return False
    if not local_origin(headers.get("Origin")):
        return False
    # a page of another site loading our address, e.g. <img src=".../camera.mjpg"> (sends no Origin). 127.0.0.1
    # with another port is "same-site", so the app's own pages pass; a page on localhost loading 127.0.0.1 is
    # "cross-site" but names a local page as where it came from
    if (headers.get("Sec-Fetch-Site") or "").lower() == "cross-site":
        source = headers.get("Origin") or headers.get("Referer") or ""
        m = re.match(r"^http://([^/]+)", str(source).strip(), re.I)
        return bool(m) and local_host(m.group(1))
    return True


# sent with every page: no framing by other sites, no guessing file types
SECURITY_HEADERS = {
    "Content-Security-Policy": "frame-ancestors 'self'",
    "X-Content-Type-Options": "nosniff",
}
