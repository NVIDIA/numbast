# Build the documentation

Use Pixi from the repository root (recommended):

1. Install one environment (first time only): `pixi install -e test-cu13` (or `test-cu12`).
2. Ensure the version is included in `docs/nv-versions.json`.
3. Build the docs with `pixi run -e test-cu13 build-docs`.
4. The HTML artifacts will be under both `docs/build/html/latest` and `docs/build/html/<version>`.

Build only the latest version with:

```bash
pixi run -e test-cu13 build-docs -- latest-only
```

If you are already in a prepared shell environment, you can still run `./build_docs.sh`
(or `./build_docs.sh latest-only`) from `docs/`.

To publish docs, keep older `docs/build/html/<version>` directories intact for the version switcher.
