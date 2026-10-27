#!/usr/bin/env bash
# 格式化或检查项目源文件，不扫描构建目录和第三方代码。
set -euo pipefail

projectRoot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
formatter="${CLANG_FORMAT:-clang-format}"
mode="${1:---write}"
if [[ $# -gt 1 || ( "$mode" != --write && "$mode" != --check ) ]]; then
    printf 'Usage: bash scripts/format.sh [--write|--check]\n' >&2
    exit 2
fi
if ! command -v "$formatter" >/dev/null 2>&1; then
    printf 'clang-format is required. Set CLANG_FORMAT to its executable path if needed.\n' >&2
    exit 1
fi

cd "$projectRoot"
mapfile -d '' -t sources < <(find src include test example -type f \
    \( -name '*.cu' -o -name '*.cuh' -o -name '*.cpp' \) -print0 | sort -z)
if [[ ${#sources[@]} -eq 0 ]]; then
    exit 0
fi
if [[ "$mode" == --check ]]; then
    "$formatter" --style=file --dry-run --Werror "${sources[@]}"
else
    "$formatter" --style=file -i "${sources[@]}"
fi
