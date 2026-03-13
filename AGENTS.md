# AGENTS

## Skill: Publish a New Numbast Mainline Version

Use this skill when asked to publish a new version of the Numbast agent from `main`.

### Required Inputs

- `NEW_VERSION` (for example, `0.6.1`)
- `PREVIOUS_VERSION` (for example, `0.6.0`)

### Steps

1. Make sure local `main` is up to date:
   - `git checkout main && git pull`
2. Create a version-bump branch:
   - `git checkout -b bump-version-NEW_VERSION`
3. Update the repo-root `VERSION` file to `NEW_VERSION`, then stage and commit:
   - `echo "NEW_VERSION" > VERSION`
   - `git add VERSION && git commit -m "chore: bump VERSION to NEW_VERSION"`
4. Push the branch:
   - `git push -u origin bump-version-NEW_VERSION`
5. Generate and save a short changelog:
   - `git log v<PREVIOUS_VERSION>..HEAD --pretty=format:"- %s" > /tmp/numbast-vNEW_VERSION-changelog.txt`
6. Open a PR from `bump-version-NEW_VERSION` and paste the changelog in the PR description.
7. Wait until that PR is merged into `main`.
8. After merge, refresh local `main`:
   - `git checkout main && git pull`
9. Create the tag annotation file non-interactively:
   - `printf "v<NEW_VERSION>\n\n" > /tmp/numbast-vNEW_VERSION-tag.txt`
   - `cat /tmp/numbast-vNEW_VERSION-changelog.txt >> /tmp/numbast-vNEW_VERSION-tag.txt`
10. Create the annotated tag without opening an editor:
   - `git tag -a v<NEW_VERSION> -F /tmp/numbast-vNEW_VERSION-tag.txt`
11. Verify the tag annotation:
   - `git show v<NEW_VERSION>`
12. Push the tag:
   - `git push origin v<NEW_VERSION>`
   - Replace `origin` with your fork or upstream remote if needed.

### Guardrails

- Do not tag a release before the `VERSION` bump PR is merged.
- Keep the PR changelog and tag annotation text consistent.
- Use the `v<NEW_VERSION>` tag format exactly.
- Use non-interactive tag creation (`-F`) so automation does not hang.

## Skill: Publish a Numbast Patch Release Line

Use this skill when asked to release a patch version from a previous tag (for example, from `v0.6.0` to `v0.6.1`).

### Required Inputs

- `NEW_VERSION` (for example, `0.6.1`)
- `PREVIOUS_VERSION` (for example, `0.6.0`)
- `PATCH_BRANCH` (for example, `0.6.x-patch`)

### Steps

1. Create and push a maintenance branch from the previous tag:
   - `git checkout v<PREVIOUS_VERSION>`
   - `git checkout -b PATCH_BRANCH`
   - `git push -u origin PATCH_BRANCH`
2. For each patch fix, use a short-lived branch based on `PATCH_BRANCH`, open a PR targeting `PATCH_BRANCH`, then wait for CI and merge.
3. Create a version bump PR into `PATCH_BRANCH`:
   - `git checkout PATCH_BRANCH && git pull`
   - `git checkout -b bump-version-NEW_VERSION`
   - `echo "NEW_VERSION" > VERSION`
   - `git add VERSION && git commit -m "Bump Version to NEW_VERSION"`
   - `git push -u origin bump-version-NEW_VERSION`
   - Open a PR from `bump-version-NEW_VERSION` into `PATCH_BRANCH`.
4. Wait until the version-bump PR is merged into `PATCH_BRANCH`.
5. Refresh local `PATCH_BRANCH`:
   - `git checkout PATCH_BRANCH && git pull`
6. Generate and save the changelog for the patch line:
   - `git log v<PREVIOUS_VERSION>..HEAD --pretty=format:"- %s" > /tmp/numbast-vNEW_VERSION-changelog.txt`
7. Create the annotated tag file non-interactively:
   - `printf "v<NEW_VERSION>\n\n" > /tmp/numbast-vNEW_VERSION-tag.txt`
   - `cat /tmp/numbast-vNEW_VERSION-changelog.txt >> /tmp/numbast-vNEW_VERSION-tag.txt`
8. Create and push the annotated tag:
   - `git tag -a v<NEW_VERSION> -F /tmp/numbast-vNEW_VERSION-tag.txt`
   - `git show v<NEW_VERSION>`
   - `git push origin v<NEW_VERSION>`

### Guardrails

- Do not use maintenance branch names starting with `v` (for example, avoid `v0.6.x-*`).
- Do not tag before the version bump PR is merged into `PATCH_BRANCH`.
- Tag the tip of `PATCH_BRANCH`, not `main`, for patch releases.
- Keep the patch PR changelog and tag annotation text consistent.
