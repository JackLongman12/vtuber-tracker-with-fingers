"""Publish a new version: build the app and put it on GitHub as a release. Installed apps offer it as an update on
their next start (avatarstudio/core/updater.py).

  .venv\\Scripts\\python.exe tools/release.py 1.2.0 --notes "What's new (shown in the app's update dialog)"
  .venv\\Scripts\\python.exe tools/release.py 1.2.0 --notes-file notes.md --dry-run   (build only, upload nothing)
  .venv\\Scripts\\python.exe tools/release.py --models    (models/ as models.zip on the "models" release: fetch_models.py)
  .venv\\Scripts\\python.exe tools/release.py 1.1.0 --prerelease --branch v1.1-experimental --notes "..."
      (experimental: a pre-release built from a branch; apps never update to it by themselves)

Before the first release: set GITHUB_REPO ("owner/repository") in avatarstudio/version.py, push your code (the tag
goes on the default branch's newest commit), and have a token: GITHUB_TOKEN=... (a fine-grained token with
"Contents: read and write" on the repository) or a logged-in GitHub CLI (`gh auth token` is used then).

Steps: VERSION in avatarstudio/version.py is set to the new version, Distributions/AvatarStudio.exe is built
(tools/build_dist.py), SHA256SUMS.txt is written next to it, then the release v<version> is created and both files
are uploaded (the app checks the download against SHA256SUMS.txt). Commit and push the changed version.py too.
Small updates: a patch from the latest release before this one (AvatarStudio-from-<that version>.avspatch,
avatarstudio/core/delta.py) is made and uploaded too, with its line in SHA256SUMS.txt; apps on that version
download it instead of the whole exe. That release's exe is kept in build_tmp/release_base (taken from
Distributions before the build when it's the same file, else downloaded once).
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from avatarstudio.core import delta  # noqa: E402
BASE_DIR = os.path.join(ROOT, "release_base")  # (not in build_tmp: the build empties that)
VERSION_FILE = os.path.join(ROOT, "avatarstudio", "version.py")
APP = os.path.join(ROOT, "Distributions", "AvatarStudio.exe")
API = "https://api.github.com"


def read_setting(name):
    m = re.search(rf'^{name} = "(.*)"$', open(VERSION_FILE, encoding="utf-8").read(), re.M)
    return m.group(1) if m else ""


def set_version(version):
    text = open(VERSION_FILE, encoding="utf-8").read()
    open(VERSION_FILE, "w", encoding="utf-8", newline="").write(
        re.sub(r'^VERSION = ".*"$', f'VERSION = "{version}"', text, flags=re.M))


def get_token():
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token
    try:
        return subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        raise SystemExit("no GitHub token: set GITHUB_TOKEN, or log in with the GitHub CLI (gh auth login)")


def delta_name(version):
    from avatarstudio.core.updater import PATCH
    return PATCH.format(version)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def call(method, url, token, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "User-Agent": "AvatarStudio-release",
        **({"Content-Type": "application/json"} if data else {})})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            text = r.read()
            return json.loads(text) if text else {}
    except urllib.error.HTTPError as e:
        if e.code == 404 and method == "GET":
            return None
        raise SystemExit(f"GitHub said {e.code} to {method} {url}: {e.read().decode('utf-8', 'replace')[:300]}")


def upload(release, path, name, token):
    url = release["upload_url"].split("{")[0] + "?name=" + urllib.parse.quote(name)
    size = os.path.getsize(path)
    print(f"uploading {name} ({size / 2 ** 20:.0f} MB)...", flush=True)
    with open(path, "rb") as f:
        req = urllib.request.Request(url, data=f, method="POST", headers={
            "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
            "User-Agent": "AvatarStudio-release", "Content-Type": "application/octet-stream",
            "Content-Length": str(size)})
        with urllib.request.urlopen(req, timeout=3600) as r:
            r.read()


def _get(url, token):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "AvatarStudio-release",
               **({"Authorization": f"Bearer {token}"} if token else {})}
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=600)


def previous_base(args, repo, token):
    """The latest release before this one, for the patch: (version, its exe here) or None. Called before the build,
    which replaces Distributions/AvatarStudio.exe."""
    if args.prerelease or not repo:
        return None
    try:
        with _get(f"{args.api}/repos/{repo}/releases/latest", token) as r:
            rel = json.load(r)
        assets = {a["name"]: a for a in rel.get("assets", [])}
        version = rel["tag_name"].lstrip("vV")
        with _get(assets["SHA256SUMS.txt"]["browser_download_url"], None) as r:
            sums = r.read().decode("utf-8", "replace")
        want = next(l.split()[0].lower() for l in sums.splitlines() if l.split()[-1:] == ["AvatarStudio.exe"])
        base = os.path.join(BASE_DIR, f"AvatarStudio-{version}.exe")
        os.makedirs(BASE_DIR, exist_ok=True)
        if os.path.exists(base) and sha256(base) == want:
            return version, base
        app = args.exe or APP
        if os.path.exists(app) and sha256(app) == want:
            shutil.copyfile(app, base)
        else:
            print(f"downloading v{version}'s app for the patch (once)...", flush=True)
            with _get(assets["AvatarStudio.exe"]["browser_download_url"], None) as r, open(base, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
        if sha256(base) != want:
            raise ValueError("it doesn't match its checksum")
        for old in os.listdir(BASE_DIR):  # (only the newest one is needed)
            if old != os.path.basename(base):
                os.remove(os.path.join(BASE_DIR, old))
        return version, base
    except Exception as e:
        print(f"no patch for small updates this time: couldn't get the previous release ({e})", flush=True)
        return None


def release_app(args, repo, token):
    version = args.version
    if not re.fullmatch(r"\d+\.\d+\.\d+", version or ""):
        raise SystemExit("give the new version as three numbers, e.g. 1.2.0")
    current = read_setting("VERSION")
    if tuple(map(int, version.split("."))) <= tuple(map(int, current.split("."))) and not args.skip_build:
        raise SystemExit(f"{version} isn't newer than this code's {current}: installed apps wouldn't update to it")
    notes = open(args.notes_file, encoding="utf-8").read() if args.notes_file else (args.notes or "")
    prev = previous_base(args, repo, token)
    if not args.skip_build:
        set_version(version)
        print(f"version {current} -> {version}; building (a few minutes)...", flush=True)
        subprocess.run([sys.executable, os.path.join(ROOT, "tools", "build_dist.py")], check=True)
    app = args.exe or APP
    sums = os.path.join(os.path.dirname(app), "SHA256SUMS.txt")
    patch = None
    if prev:
        patch = os.path.join(BASE_DIR, delta_name(prev[0]))
        size, _ = delta.make(prev[1], app, patch)
        print(f"patch from v{prev[0]}: {size / 2 ** 20:.0f} MB", flush=True)
        if size > 0.6 * os.path.getsize(app):  # (hardly smaller: not worth a second way to update)
            patch = None
    with open(sums, "w", newline="\n") as f:
        f.write(f"{sha256(app)}  AvatarStudio.exe\n")
        if patch:
            f.write(f"{sha256(patch)}  {os.path.basename(patch)}\n")
    print(f"{app}: {os.path.getsize(app) / 2 ** 20:.0f} MB, checksum in {sums}")
    if args.dry_run:
        print("dry run: nothing uploaded")
        return
    body = {"tag_name": f"v{version}", "name": f"Avatar Studio {version}" + (" (experimental)" if args.prerelease else ""),
            "body": notes, "draft": False, "prerelease": bool(args.prerelease)}
    if args.prerelease:
        body["make_latest"] = "false"
        if args.branch:  # (tagged on that branch's newest commit, not the default branch's)
            body["target_commitish"] = args.branch
    rel = call("POST", f"{args.api}/repos/{repo}/releases", token, body)
    upload(rel, app, "AvatarStudio.exe", token)
    upload(rel, sums, "SHA256SUMS.txt", token)
    if patch:
        upload(rel, patch, os.path.basename(patch), token)
    print(f"released: {rel.get('html_url')}\n" + ("A pre-release: installed apps don't offer it (they only take the "
                                                    "latest full release); people download it themselves."
                                                    if args.prerelease else "Installed apps offer it on their next start."))
    if not args.skip_build:
        print(f"Commit and push avatarstudio/version.py (now {version}).")


def release_models(args, repo, token):
    """models/ as models.zip on a release tagged "models" (a pre-release, so apps never take it for an update),
    with the models' licences (packaging/licenses, THIRD_PARTY_NOTICES.md)."""
    os.makedirs(os.path.join(ROOT, "build_tmp"), exist_ok=True)
    path = os.path.join(ROOT, "build_tmp", "models.zip")
    lic_dir = os.path.join(ROOT, "packaging", "licenses")
    licenses = {name: os.path.join(lic_dir, name) for name in os.listdir(lic_dir)}
    licenses["THIRD_PARTY_NOTICES.md"] = os.path.join(ROOT, "THIRD_PARTY_NOTICES.md")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for name in sorted(os.listdir(os.path.join(ROOT, "models"))):
            if name not in licenses and os.path.isfile(os.path.join(ROOT, "models", name)):  # (fetched copies)
                z.write(os.path.join(ROOT, "models", name), name)
        for name, src in sorted(licenses.items()):
            z.write(src, name)
    rel = call("GET", f"{args.api}/repos/{repo}/releases/tags/models", token) or call(
        "POST", f"{args.api}/repos/{repo}/releases", token, {
            "tag_name": "models", "name": "Model files", "prerelease": True,
            "body": "The model files the app needs (tools/fetch_models.py downloads them into models/)."})
    for asset in rel.get("assets", []):
        if asset["name"] == "models.zip":
            call("DELETE", f"{args.api}/repos/{repo}/releases/assets/{asset['id']}", token)
    upload(rel, path, "models.zip", token)
    print(f"models.zip uploaded: {rel.get('html_url')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version", nargs="?", help="the new version, e.g. 1.2.0")
    ap.add_argument("--notes", help="what's new (Markdown; shown in the app's update dialog)")
    ap.add_argument("--notes-file", help="the notes from a file")
    ap.add_argument("--dry-run", action="store_true", help="build and write the checksum, upload nothing")
    ap.add_argument("--skip-build", action="store_true", help="publish the existing Distributions/AvatarStudio.exe")
    ap.add_argument("--models", action="store_true", help="upload models/ as models.zip (for tools/fetch_models.py)")
    ap.add_argument("--prerelease", action="store_true",
                    help="an experimental release: not offered as an update, and not the download link's 'latest'")
    ap.add_argument("--branch", help="with --prerelease: tag this branch's newest commit (e.g. v1.1-experimental)")
    ap.add_argument("--exe", help=argparse.SUPPRESS)  # (checks: another file)
    ap.add_argument("--api", default=API, help=argparse.SUPPRESS)  # (checks: a stand-in GitHub)
    ap.add_argument("--repo", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    repo = args.repo or read_setting("GITHUB_REPO")
    if not repo and not args.dry_run:
        raise SystemExit('set GITHUB_REPO = "owner/repository" in avatarstudio/version.py first')
    token = None if args.dry_run else get_token()
    if args.models:
        release_models(args, repo, token)
    else:
        release_app(args, repo, token)


if __name__ == "__main__":
    main()
