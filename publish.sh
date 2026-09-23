#!/usr/bin/env bash
# Rebuild the hub page, commit, and push to GitHub Pages.
# Usage: ./publish.sh [folder]   (folder is only used for the commit message and the printed link)
#        ./publish.sh papers     unattended mode (launchd): stages only paper-*, index.html and hub.json
set -euo pipefail
cd "$(dirname "$0")"

unattended=0
[ "${1:-}" = "papers" ] && unattended=1

branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$branch" != "main" ]; then
  echo "Refusing to publish from branch '$branch' (GitHub Pages serves main)."
  [ "$unattended" -eq 1 ] && exit 0
  exit 1
fi

lock="$(git rev-parse --git-dir)/publish.lock"
if [ -e "$lock" ] && find "$lock" -maxdepth 0 -mmin +60 | grep -q .; then
  echo "Removing stale publish lock."
  rmdir "$lock"
fi
if ! mkdir "$lock" 2>/dev/null; then
  echo "Another publish is running."
  exit 0
fi
trap 'rmdir "$lock"' EXIT

if [ "$unattended" -eq 1 ]; then
  paths=(':(glob)paper-*/**' index.html hub.json)
  node sync-papers.mjs
  node course-nav.mjs
  git add -A -- ':(glob)paper-*/**'
  node build-hub.mjs --tracked
else
  paths=(.)
  node sync-papers.mjs || echo "papers: sync failed, publishing without paper changes"
  node course-nav.mjs
  node build-hub.mjs
fi
git add -A -- "${paths[@]}"

committed=0
if ! git diff --cached --quiet -- "${paths[@]}"; then
  git commit -q -m "Publish ${1:-explainers}" -- "${paths[@]}"
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
if [ -n "${1:-}" ] && [ -d "$1" ] && [ ! -f "$1/index.html" ]; then page=$(cd "$1" && ls *.html | head -1); fi
echo "Pushed. Live in about a minute at https://${repo%%/*}.github.io/${repo#*/}/$( [ -d "${1:-}" ] && echo "$1/" )$page"
