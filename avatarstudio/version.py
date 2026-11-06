"""This build's version, and where updates come from.

tools/release.py sets VERSION when it publishes a release; the app compares it with the newest release on GitHub
(avatarstudio/core/updater.py) and offers the update.
"""
VERSION = "1.1.0"

# "owner/repository" on GitHub whose releases the app updates from; empty = no update checks
GITHUB_REPO = "JackLongman12/vtuber-tracker-with-fingers"
