# Install

Prebuilt Conda packages are available. For development, we recommend installing from source.

Install via Conda:

```bash
conda install Numbast
```

From source (recommended for development):

Numbast depends on LLVM's libClangTooling. For an easy development setup, we recommend installing `clangdev` via Conda. A recommended `clangdev` entry is already included in `conda/environment.yaml`, so developing inside this Conda environment saves you from installing LLVM manually.

```bash
# from repo root
conda env create -f conda/environment.yaml
conda activate numbast

# build the C++ header parser
bash ast_canopy/build.sh

# install the Numbast Python package
pip install numbast/
```

Validate the installation (optional):

```bash
pytest ast_canopy/ numbast/
```

## Building Documentation

### Dependencies

Use Conda to ensure consistent versions. From `conda/environment.yaml`, the doc-related dependencies are:

- sphinx
- myst-nb
- sphinx-copybutton

You also need Python and a working environment to import `numbast` if you want versioned builds to reflect the installed package version (optional).

### Build Steps

- Build all versions listed in `docs/versions.json`:

```bash
cd docs
./build_docs.sh
```

- Build only the latest version:

```bash
cd docs
./build_docs.sh latest-only
```

Artifacts are generated under:
- `docs/build/html/latest`
- `docs/build/html/<version>` where `<version>` comes from `SPHINX_NUMBAST_VER` or detected package/version files.

Notes:
- The build script sets `SPHINX_NUMBAST_VER` automatically from the installed `numbast` package version, `numbast/VERSION`, or top-level `VERSION`. If none is found, it uses `latest`.
- Output also copies `versions.json` and creates a redirect `index.html` for convenience.
