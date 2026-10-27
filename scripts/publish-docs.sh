#!/usr/bin/env bash
# 将已生成的 HTML 同步到文档仓库，保留发布分支历史。
set -euo pipefail

projectRoot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repository="${DOCS_REPOSITORY:-git@github.com:Mgepahmge/ThermoFEM-Doc.git}"
branch="${DOCS_BRANCH:-gh-pages}"
htmlDirectory="$projectRoot/build/docs/html"
if [[ ! -f "$htmlDirectory/index.html" ]]; then
    printf 'Build documentation before publishing.\n' >&2
    exit 1
fi
git check-ref-format --branch "$branch" > /dev/null
publishDirectory="$(mktemp -d)"
trap 'rm -rf -- "$publishDirectory"' EXIT

git init --quiet --initial-branch="$branch" "$publishDirectory"
cd "$publishDirectory"
git remote add origin "$repository"
# 先验证远程访问，避免将认证失败误判为尚无发布分支。
remoteHead="$(git ls-remote --heads origin "refs/heads/$branch")"
if [[ -n "$remoteHead" ]]; then
    git fetch --quiet --depth=1 origin "refs/heads/$branch"
    git reset --quiet --hard FETCH_HEAD
fi
rsync -a --delete --exclude='.git' "$htmlDirectory/" "$publishDirectory/"
touch .nojekyll
git add --all
if git diff --cached --quiet; then
    printf 'Documentation is unchanged.\n'
    exit 0
fi
git -c user.name='github-actions[bot]' \
    -c user.email='41898282+github-actions[bot]@users.noreply.github.com' \
    commit --quiet -m 'docs(site)：更新项目文档'
git push origin "HEAD:refs/heads/$branch"
