"""Get the model files into models/ after cloning the repository (they're too big for git; tools/release.py --models
puts them on the repository's "models" release).

  .venv\\Scripts\\python.exe tools/fetch_models.py
"""
import os
import re
import sys
import tempfile
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    text = open(os.path.join(ROOT, "avatarstudio", "version.py"), encoding="utf-8").read()
    repo = (re.search(r'^GITHUB_REPO = "(.*)"$', text, re.M) or [None, ""])[1]
    if not repo:
        raise SystemExit('set GITHUB_REPO = "owner/repository" in avatarstudio/version.py first')
    url = f"https://github.com/{repo}/releases/download/models/models.zip"
    tmp = os.path.join(tempfile.gettempdir(), "avatarstudio-models.zip")
    print(f"downloading {url} ...", flush=True)
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total, done = int(r.headers.get("Content-Length") or 0), 0
        for chunk in iter(lambda: r.read(1 << 20), b""):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r{done / 2 ** 20:.0f} / {total / 2 ** 20:.0f} MB", end="", flush=True)
    print()
    out = os.path.join(ROOT, "models")
    os.makedirs(out, exist_ok=True)
    with zipfile.ZipFile(tmp) as z:
        for name in z.namelist():
            if name.endswith("/") or os.path.basename(name) in ("", ".", ".."):
                continue
            with z.open(name) as src, open(os.path.join(out, os.path.basename(name)), "wb") as dst:  # (no paths)
                dst.write(src.read())
            print(f"  models/{os.path.basename(name)}")
    os.remove(tmp)


if __name__ == "__main__":
    sys.exit(main())
