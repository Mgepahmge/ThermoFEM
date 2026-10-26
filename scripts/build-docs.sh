#!/usr/bin/env bash
# 从项目根目录运行 Doxygen，生成项目接口文档。
set -euo pipefail

projectRoot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if ! command -v doxygen >/dev/null 2>&1; then
    printf 'Doxygen is required. Install doxygen and retry.\n' >&2
    exit 1
fi
cd "$projectRoot"
mkdir -p build/docs
doxygen docs/Doxyfile
printf 'Documentation: %s/build/docs/html/index.html\n' "$projectRoot"
