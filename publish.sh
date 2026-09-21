#!/usr/bin/env bash
# Rebuild the hub page, commit, and push to GitHub Pages.
# Usage: ./publish.sh [folder]   (folder is only used for the commit message and the printed link)
set -euo pipefail
cd "$(dirname "$0")"

node sync-papers.mjs
node build-hub.mjs
git add -A
if git diff --cached --quiet; then
  echo "Nothing new to publish."
  exit 0
fi
git commit -q -m "Publish ${1:-explainers}"
git push -q

repo=$(git remote get-url origin | sed -E 's#(git@github.com:|https://github.com/)##; s#\.git$##')
page=""
if [ -n "${1:-}" ] && [ ! -f "$1/index.html" ]; then page=$(cd "$1" && ls *.html | head -1); fi
echo "Pushed. Live in about a minute at https://${repo%%/*}.github.io/${repo#*/}/${1:+$1/}$page"
