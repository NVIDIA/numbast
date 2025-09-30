# Build the documentation

1. Install the `numbast` package version you want to document (optional).
2. Ensure the version is included in `versions.json`.
3. Build the docs with `./build_docs.sh`.
4. The HTML artifacts will be under both `./build/html/latest` and `./build/html/<version>`.

You can build only the latest version with:

```bash
./build_docs.sh latest-only
```

To publish docs, keep older `build/html/<version>` directories intact for the version switcher.
