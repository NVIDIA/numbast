# AGENTS

## Skill: Publish a New Numbast Agent Version

Use this skill when asked to publish a new version of the Numbast agent.

### Required Inputs

- `NEW_VERSION` (for example, `0.6.1`)
- `PREVIOUS_VERSION` (for example, `0.6.0`)

### Steps

1. Make sure local `main` is up to date:
   - `git checkout main && git pull`
2. Update the repo-root `VERSION` file to `NEW_VERSION`.
3. Generate a short changelog:
   - `git log v<PREVIOUS_VERSION>..HEAD --oneline --pretty=format:"- %s"`
4. Open a PR for the `VERSION` update and paste the changelog in the PR description.
5. Wait until that PR is merged into `main`.
6. After merge, refresh local `main` and create the release tag:
   - `git checkout main && git pull`
   - `git tag -a v<NEW_VERSION>`
7. In the tag annotation, paste the same changelog used in the PR description, using:

   ```text
   v<NEW_VERSION>

   - ... (bullet points on release items)
   ```

8. Push the tag:
   - `git push git@github.com:NVIDIA/numbast.git v<NEW_VERSION>`

### Guardrails

- Do not tag a release before the `VERSION` bump PR is merged.
- Keep the PR changelog and tag annotation text consistent.
- Use the `v<NEW_VERSION>` tag format exactly.
