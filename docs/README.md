# Build the documentation

1. Install the `numbast` package of the version that we need to document.
2. Ensure the version is included in [`nv-versions.json`](./nv-versions.json) and [`versions.json`](./versions.json).
3. Build the docs with `./build_docs.sh`.
4. The html artifacts should be available under both `./build/html/latest` and `./build/html/<version>`.

You can build only the latest version with:

```bash
./build_docs.sh latest-only
```

To publish the docs with the built version, it is important to note that the html files of older versions
should be kept intact, in order for the version selection (through `nv-versions.json`) to work.
