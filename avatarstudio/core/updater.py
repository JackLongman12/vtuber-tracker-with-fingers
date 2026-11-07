"""Updates through the app, from the app's GitHub releases (avatarstudio/version.py: VERSION, GITHUB_REPO).

  check()      GitHub's latest release of GITHUB_REPO. Newer than VERSION (tag v1.2.0 > 1.0.0) and it has this OS's
               app file (AvatarStudio.exe on Windows) -> the update's details, else None.
  download()   that file, into a file next to the running app, checked against the release's SHA256SUMS.txt
               (a release without it, or a file that doesn't match, isn't installed).
  install()    swaps it in: the running one-file AvatarStudio.exe is renamed to AvatarStudio.old.exe (Windows lets
               a running program be renamed, not replaced), the new file takes its name and is started; the caller
               then quits. The next start removes the old file (cleanup()). The new version unpacks itself on its
               first start like any new build.

Releases are published with tools/release.py. Only packaged builds update (development runs use git).
"""
import hashlib
import json
import os
import subprocess
import sys
import urllib.request

from avatarstudio.version import GITHUB_REPO, VERSION

API = "https://api.github.com"
APP_FILE = {"win32": "AvatarStudio.exe"}  # (this OS's app file in a release; macOS / Linux builds: not yet)
SUMS = "SHA256SUMS.txt"


def parse_version(v):
    """'v1.2.10' -> (1, 2, 10): versions compare as numbers, not text."""
    parts = []
    for p in str(v).strip().lstrip("vV").split("."):
        digits = "".join(c for c in p if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts + [0] * (3 - len(parts)))


def _open(url, timeout=15):
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                               "User-Agent": f"AvatarStudio/{VERSION}"})
    return urllib.request.urlopen(req, timeout=timeout)


def check(repo=GITHUB_REPO, current=VERSION, api=API):
    """The newest release if it's newer than `current` and has this OS's app file, else None."""
    if not repo or sys.platform not in APP_FILE:
        return None
    with _open(f"{api}/repos/{repo}/releases/latest") as r:
        rel = json.load(r)
    tag = rel.get("tag_name", "")
    if parse_version(tag) <= parse_version(current):
        return None
    assets = {a["name"]: a for a in rel.get("assets", [])}
    app = assets.get(APP_FILE[sys.platform])
    if not app:
        return None
    return {"version": tag.lstrip("vV"), "notes": rel.get("body") or "", "page": rel.get("html_url", ""),
            "url": app["browser_download_url"], "size": app.get("size", 0), "name": app["name"],
            "sums_url": assets[SUMS]["browser_download_url"] if SUMS in assets else None}


def download(info, folder, progress=None):
    """The update's app file into `folder` (next to the running app, so the swap is a rename on the same drive),
    checked against SHA256SUMS.txt. progress(done_bytes, total_bytes). -> its path."""
    if not info.get("sums_url"):
        raise ValueError("the release has no SHA256SUMS.txt, so the download can't be checked")
    home = f"https://github.com/{GITHUB_REPO}/releases/download/"
    if not all(str(info.get(k, "")).startswith(home) for k in ("url", "sums_url")):
        raise ValueError("the update isn't on Avatar Studio's own GitHub releases; nothing was downloaded")
    with _open(info["sums_url"]) as r:
        sums = r.read().decode("utf-8", "replace")
    expected = next((line.split()[0].lower() for line in sums.splitlines()
                     if len(line.split()) >= 2 and line.split()[-1].lstrip("*") == info["name"]), None)
    if not expected:
        raise ValueError(f"SHA256SUMS.txt has no line for {info['name']}")
    path = os.path.join(folder, info["name"] + ".download")
    digest, done = hashlib.sha256(), 0
    try:
        with _open(info["url"], timeout=60) as r, open(path, "wb") as f:
            total = int(r.headers.get("Content-Length") or info.get("size") or 0)
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
    except BaseException:
        _remove(path)
        raise
    if digest.hexdigest() != expected:
        _remove(path)
        raise ValueError("the download doesn't match its checksum (damaged or changed); nothing was installed")
    return path


def install(downloaded, app):
    """Swap the downloaded file in for `app` (the running one-file AvatarStudio.exe) and start it."""
    old = os.path.splitext(app)[0] + ".old.exe"
    _remove(old)  # (left from an update before)
    os.replace(app, old)  # (renaming the running program is allowed)
    try:
        os.replace(downloaded, app)
    except OSError:
        os.replace(old, app)  # (put it back as it was)
        raise
    flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP) if sys.platform == "win32" else 0
    subprocess.Popen([app], creationflags=flags, close_fds=True, cwd=os.path.dirname(app))


def cleanup(app):
    """After an update: the previous version's file, renamed by install(), goes."""
    if app:
        _remove(os.path.splitext(app)[0] + ".old.exe")
        _remove(app + ".download")


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass
