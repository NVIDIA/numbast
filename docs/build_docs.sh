#!/bin/bash

set -ex

if [[ "$#" == "0" ]]; then
    LATEST_ONLY="0"
elif [[ "$#" == "1" && "$1" == "latest-only" ]]; then
    LATEST_ONLY="1"
else
    echo "usage: ./build_docs.sh [latest-only]"
    exit 1
fi

# SPHINX_NUMBAST_VER is used to create a subdir under build/html
# If there's a post release (ex: .post1) we don't want it to show up in the
# version selector or directory structure.
if [[ -z "${SPHINX_NUMBAST_VER}" ]]; then
    set +e
    VER_FROM_PKG=$(python - <<'PY'
from importlib.metadata import version, PackageNotFoundError
try:
    ver = version('numbast')
    print('.'.join(str(ver).split('.')[:3]).split('+')[0])
except PackageNotFoundError:
    print('')
PY
    )
    set -e
    if [[ -n "${VER_FROM_PKG}" ]]; then
        export SPHINX_NUMBAST_VER=${VER_FROM_PKG}
    elif [[ -f ../numbast/VERSION ]]; then
        export SPHINX_NUMBAST_VER=$(head -n1 ../numbast/VERSION | awk -F'+' '{print $1}')
    elif [[ -f ../VERSION ]]; then
        export SPHINX_NUMBAST_VER=$(head -n1 ../VERSION | awk -F'+' '{print $1}')
    else
        export SPHINX_NUMBAST_VER=latest
    fi
fi

# auto-clean previous artifacts for a consistent rebuild
rm -rf build/.doctrees
rm -rf build/html/${SPHINX_NUMBAST_VER}
rm -rf build/html/latest

# build the docs (in parallel)
SPHINXOPTS="-j 4 -d build/.doctrees" make html

# For debugging/developing (conf.py), you can build in serial to avoid obscure Sphinx errors
#SPHINXOPTS="-v" make html

# to support version dropdown menu
cp ./nv-versions.json build/html

# to have a redirection page (to the latest docs)
cp source/_templates/main.html build/html/index.html

# ensure that the latest docs is the one we built
if [[ $LATEST_ONLY == "0" ]]; then
    cp -r build/html/${SPHINX_NUMBAST_VER} build/html/latest
else
    mv build/html/${SPHINX_NUMBAST_VER} build/html/latest
fi

# ensure that the Sphinx reference uses the latest docs
cp build/html/latest/objects.inv build/html || true
