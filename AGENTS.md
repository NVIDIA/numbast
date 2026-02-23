# AGENTS

## Skill: Publish a New Numbast Agent Version

Use this skill when asked to publish a new version of the Numbast agent.

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
