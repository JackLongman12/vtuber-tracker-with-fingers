"""Report a problem (Settings > Report a problem): a bug report you read first, then send yourself as a GitHub issue.

Nothing is sent by the app. It puts together what helps find a bug - your description, the app version, the OS, the
computer's CPU / memory / GPU runtime, your settings (switches and sliders only) and the last lines of the app's logs
- with personal things taken out (your user name, the computer's name, folder paths under your home folder, email
and network addresses). You see the whole text; "Open GitHub" opens a new issue with it filled in, in your browser,
under your own GitHub account, and nothing goes anywhere until you press Submit there.
"""
import os
import platform
import re
import sys
import urllib.parse

from avatarstudio.core.paths import user_file
from avatarstudio.version import GITHUB_REPO, VERSION

LOG_LINES = 60
URL_MAX = 7500  # (GitHub refuses much longer new-issue links; the full text can still be copied)
NOISE = re.compile(r"^([WIEF]\d{4} \d|WARNING: All log messages before absl|WARNING: Logging before InitGoogle|"
                   r"INFO: Created TensorFlow Lite)")


def scrub(text):
    """Personal things out: home folder paths, user and computer name, email and network addresses."""
    home = os.path.expanduser("~")
    for h in {home, home.replace("\\", "/"), home.replace("\\", "\\\\")}:
        if len(h) > 3:
            text = re.sub(re.escape(h), "~", text, flags=re.I)
    for name in {os.environ.get("USERNAME"), os.environ.get("USER"), os.environ.get("COMPUTERNAME"), platform.node()}:
        if name and len(name) >= 3:
            text = re.sub(rf"(?<![\w-]){re.escape(name)}(?![\w-])", "<name>", text, flags=re.I)
    text = re.sub(r"[\w.+-]+@[\w-]+(\.[\w-]+)+", "<email>", text)
    text = re.sub(r"\b(?!127\.0\.0\.1\b)(?!0\.0\.0\.0\b)\d{1,3}(\.\d{1,3}){3}\b", "<ip>", text)
    return text


def _system():
    lines = [f"Avatar Studio {VERSION} ({'app' if getattr(sys, 'frozen', False) else 'development run'})",
             f"OS: {platform.platform()}"]
    try:
        import psutil
        lines.append(f"CPU: {psutil.cpu_count(logical=False)} cores / {psutil.cpu_count()} threads, "
                     f"RAM: {psutil.virtual_memory().total / 2 ** 30:.0f} GB")
    except Exception:
        pass
    try:
        from avatarstudio.core.os_support import gpu_providers
        lines.append(f"GPU runtime: {', '.join(gpu_providers()) or 'none (CPU only)'}")
    except Exception:
        pass
    return lines


def _settings():
    try:
        from avatarstudio.core.settings import Settings
        s = Settings.load().as_dict()
        return ", ".join(f"{k}={v}" for k, v in s.items() if isinstance(v, (bool, int, float)))
    except Exception as e:
        return f"(couldn't read the settings: {e})"


def _logs():
    folder, out = user_file("logs"), []
    if not os.path.isdir(folder):
        return out
    for name in sorted(os.listdir(folder), key=lambda n: -os.path.getmtime(os.path.join(folder, n))):
        try:
            with open(os.path.join(folder, name), encoding="utf-8", errors="replace") as f:
                lines = [l.rstrip() for l in f if l.strip() and not NOISE.match(l)]
        except OSError:
            continue
        if lines:
            out.append((name, lines[-LOG_LINES:]))
    return out


def build(description=""):
    """-> {"title", "text" (the whole report), "url" (a new GitHub issue with it, or None), "short" (link shortened)}."""
    desc = (description or "").strip() or "(describe what happened, and what you expected)"
    title = scrub(desc.splitlines()[0][:80]) if description and description.strip() else "Problem report"

    def body(n_lines):
        parts = ["**What happened**", "", scrub(desc), "", "**System**", "", *[f"- {l}" for l in _system()], "",
                 "**Settings**", "", scrub(_settings()), ""]
        for name, lines in _logs():
            parts += [f"**Log: {name}** (last {min(n_lines, len(lines))} lines)", "", "```text",
                      *[scrub(l) for l in lines[-n_lines:]], "```", ""]
        parts.append("_Made by Avatar Studio's Report a problem (personal details removed)._")
        return "\n".join(parts)

    text, url, short = body(LOG_LINES), None, False
    if GITHUB_REPO:
        for n in (LOG_LINES, 30, 15, 5, 0):
            b = body(n)
            url = (f"https://github.com/{GITHUB_REPO}/issues/new?"
                   + urllib.parse.urlencode({"title": title, "body": b}))
            if len(url) <= URL_MAX:
                short = n < LOG_LINES
                break
    return {"title": title, "text": text, "url": url, "short": short}
