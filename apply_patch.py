"""
apply_patch.py  -  adds the new page reader (retry + Chrome-like fallback) to deep_oq.py

HOW TO USE (Windows):
  1. Put this file in the SAME folder as deep_oq.py
  2. Open a terminal in that folder and run:   python apply_patch.py
  3. Then upload deep_oq.py to GitHub and let Render redeploy.
A backup is saved as deep_oq.py.bak. If anything looks wrong it restores the original by itself.
"""
import ast, os, re, shutil, sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "deep_oq.py"

NEW_CODE = r'''OQ_READ_TIMEOUT = _env_int("DEEP_OQ_READ_TIMEOUT", 25)   # seconds to wait for one page (was fixed at 20)
_BLOCKED = (403, 406, 429, 503)                          # statuses sites use to turn robots away
_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.5",
    "Accept-Language": "en-IN,en;q=0.9,hi;q=0.8",
}


def _cffi_ok():
    try:
        import curl_cffi  # noqa: F401
        return True
    except Exception:
        return False


def _get_pinned(u, ip, timeout):
    """Normal safe fetch: connects to the already-checked public IP. Returns (status, content_type, location, bytes)."""
    conn = (_PinS if u.scheme == "https" else _Pin)(u.hostname, ip, u.port, timeout)
    try:
        path = (u.path or "/") + ("?" + u.query if u.query else "")
        hdr = dict(_BROWSER_HEADERS, Host=u.netloc)
        hdr["Accept-Encoding"] = "identity"
        conn.request("GET", path, headers=hdr)
        r = conn.getresponse()
        loc = r.getheader("Location") or ""
        ctype = r.getheader("Content-Type") or ""
        raw = b"" if r.status in (301, 302, 303, 307, 308) and loc else r.read(6_000_000)
        return r.status, ctype, loc, raw
    finally:
        conn.close()


def _get_cffi(u, ip, timeout):
    """Fetch that looks like real Chrome to the site (TLS fingerprint too). Needs: pip install curl_cffi."""
    from curl_cffi import requests as cr
    kw = dict(impersonate="chrome", timeout=timeout, allow_redirects=False, stream=True,
              headers={"Accept-Language": _BROWSER_HEADERS["Accept-Language"]})
    pin = None
    if ":" not in ip:  # pin the checked IP so DNS cannot be switched to a private address (IPv4 only)
        try:
            from curl_cffi import CurlOpt
            port = u.port or (443 if u.scheme == "https" else 80)
            pin = {CurlOpt.RESOLVE: ["%s:%d:%s" % (u.hostname, port, ip)]}
        except Exception:
            pin = None
    try:
        r = cr.get(u.geturl(), curl_options=pin, **kw) if pin else cr.get(u.geturl(), **kw)
    except TypeError:  # this curl_cffi version has no curl_options
        r = cr.get(u.geturl(), **kw)
    try:
        loc = r.headers.get("location") or ""
        ctype = r.headers.get("content-type") or ""
        buf = b""
        if not (r.status_code in (301, 302, 303, 307, 308) and loc):
            for chunk in r.iter_content():
                buf += chunk
                if len(buf) >= 6_000_000:
                    break
        return r.status_code, ctype, loc, buf[:6_000_000]
    finally:
        try:
            r.close()
        except Exception:
            pass


def _read_err(e):
    t = str(e).lower()
    if isinstance(e, ssl.SSLError) or "ssl" in t or "tls" in t or "handshake" in t:
        return "the site refused a secure connection from this server (it blocks automated visitors)"
    if isinstance(e, (socket.timeout, TimeoutError)) or "timed out" in t or "timeout" in t:
        return "the site took too long to answer"
    return "the site could not be reached (%s)" % str(e)[:120]


def _fetch_hop(u, ip):
    """One page request. First the normal safe way; on a timeout, SSL failure or a robot-block, try again as Chrome."""
    tries = [_get_pinned, _get_cffi if _cffi_ok() else _get_pinned]
    last, blocked = None, None
    for i, fn in enumerate(tries):
        try:
            res = fn(u, ip, OQ_READ_TIMEOUT)
        except Exception as e:
            last = e
            print("page read attempt %d failed for %s: %s" % (i + 1, u.hostname, str(e)[:140]))
            continue
        if res[0] in _BLOCKED and i < len(tries) - 1:
            blocked = res
            continue
        return res
    if blocked:
        return blocked
    raise ValueError(_read_err(last))


def read_url(url, max_chars=None):
    """Download one public web page and return (title, plain text). Follows up to 8 redirects, re-checking each hop."""
    max_chars = max_chars or OQ_PAGE_CHARS
    for _ in range(8):
        u = urllib.parse.urlsplit(url)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise ValueError("Only http and https links work.")
        ip = _public_ip(u.hostname)
        status, ctype, loc, raw = _fetch_hop(u, ip)
        if status in (301, 302, 303, 307, 308) and loc:
            url = urllib.parse.urljoin(url, loc)
            continue
        if status >= 400:
            raise ValueError("The site answered with error %d." % status)
        ctype = ctype.lower()
        if ctype and not any(t in ctype for t in ("text/", "html", "xml", "json")):
            raise ValueError("That link is not a web page (%s)." % ctype.split(";")[0])
        m = re.search(r"charset=([\w-]+)", ctype)
        try:
            txt = raw.decode(m.group(1) if m else "utf-8", "replace")
        except LookupError:
            txt = raw.decode("utf-8", "replace")
        if "html" in ctype or "<html" in txt[:2000].lower():
            ps = _Text()
            ps.feed(txt)
            body = re.sub(r"\n\s*\n+", "\n", "".join(ps.parts)).strip()
            return ps.title.strip(), body[:max_chars]
        return "", txt.strip()[:max_chars]
    raise ValueError("Too many redirects.")
'''

OLD_MSG = 'return "Could not read that page: %s" % e'
NEW_MSG = 'return "Skipped this page: %s. Try a different website." % e'
OLD_PIP = "pip install flask groq gunicorn pypdf ddgs imageio-ffmpeg anthropic sympy openai"
NEW_PIP = OLD_PIP + " curl_cffi"


def main():
    if not os.path.isfile(PATH):
        sys.exit("Cannot find %s. Put apply_patch.py in the same folder as deep_oq.py." % PATH)
    with open(PATH, encoding="utf-8", newline="") as f:
        raw = f.read()
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")

    if "def _fetch_hop(" in text:
        sys.exit("Already patched: deep_oq.py already has the new page reader. Nothing changed.")

    m = re.search(r"^def read_url\(url, max_chars=None\):[^\n]*\n", text, re.M)
    if not m:
        sys.exit("Could not find 'def read_url(url, max_chars=None):' in %s. Nothing changed." % PATH)
    nxt = re.search(r"^\S", text[m.end():], re.M)          # next line that starts at column 0
    end = m.end() + nxt.start() if nxt else len(text)
    new_text = text[:m.start()] + NEW_CODE.rstrip() + "\n\n\n" + text[end:]
    changes = ["read_url replaced"]

    if OLD_MSG in new_text:
        new_text = new_text.replace(OLD_MSG, NEW_MSG)
        changes.append("agent page-error message updated")
    if OLD_PIP in new_text and NEW_PIP not in new_text:
        new_text = new_text.replace(OLD_PIP, NEW_PIP)
        changes.append("curl_cffi added to the build-command notes")

    try:
        ast.parse(new_text)
    except SyntaxError as e:
        sys.exit("The patched file would have a syntax error (line %s). Nothing changed." % e.lineno)

    shutil.copyfile(PATH, PATH + ".bak")
    if crlf:
        new_text = new_text.replace("\n", "\r\n")
    with open(PATH, "w", encoding="utf-8", newline="") as f:
        f.write(new_text)
    print("Done. " + "; ".join(changes) + ".")
    print("Backup saved as " + PATH + ".bak")
    print("Next: add  curl_cffi  to your Render Build Command, upload deep_oq.py to GitHub, and redeploy.")


main()
