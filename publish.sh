#!/usr/bin/env bash
# Rebuild the hub page, commit, and push to GitHub Pages.
# Usage: ./publish.sh [folder]   (folder is only used for the commit message and the printed link)
set -euo pipefail
cd "$(dirname "$0")"

node sync-papers.mjs
node build-hub.mjs
git add -A

committed=0
if ! git diff --cached --quiet; then
  git commit -q -m "Publish ${1:-explainers}"
  committed=1
fi

pushed=0
if [ "$(git rev-list --count @{u}..HEAD)" -gt 0 ]; then
  git push -q
  pushed=1
fi

if [ "$committed" -eq 0 ] && [ "$pushed" -eq 0 ]; then
  echo "Nothing new to publish."
  exit 0
fi

repo=$(git remote get-url origin | sed -E 's#(git@github.com:|https://github.com/)##; s#\.git$##')
page=""
if [ -n "${1:-}" ] && [ ! -f "$1/index.html" ]; then page=$(cd "$1" && ls *.html | head -1); fi
echo "Pushed. Live in about a minute at https://${repo%%/*}.github.io/${repo#*/}/${1:+$1/}$page"
