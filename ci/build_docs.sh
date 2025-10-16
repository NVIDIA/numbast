#!/usr/bin/env bash

set -euo pipefail

# Delegate to the project docs build script. Accept optional "latest-only" arg.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${SCRIPT_DIR}/.."

pushd "${REPO_ROOT}/docs" >/dev/null
./build_docs.sh "$@"
popd >/dev/null
