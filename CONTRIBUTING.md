# Contributing to Numbast

If you are interested in contributing to Numbast, your contributions will fall
into three categories:
1. You want to report a bug, feature request, or documentation issue
    - File an [issue](https://github.com/nvidia/numbast/issues/new/choose)
    describing what you encountered or what you want to see changed.
    - Please run and paste the output of the `print_env.sh` script while
    reporting a bug to gather and report relevant environment details.
    - The team will evaluate the issues and triage them, scheduling
    them for a release. If you believe the issue needs priority attention
    comment on the issue to notify the team.
2. You want to propose a new Feature and implement it
    - Post about your intended feature, and we shall discuss the design and
    implementation.
    - Once we agree that the plan looks good, go ahead and implement it, using
    the [code contributions](#code-contributions) guide below.
3. You want to implement a feature or bug-fix for an outstanding issue
    - Follow the [code contributions](#code-contributions) guide below.
    - If you need more context on a particular issue, please ask and we shall
    provide.

## Code contributions

### Your first issue

1. Read the project's [README.md](https://github.com/nvidia/numbast/blob/main/README.md)
    to learn how to setup the development environment.
2. Find an issue to work on. The best way is to look for the [good first issue](https://github.com/nvidia/numbast/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
    or [help wanted](https://github.com/nvidia/numbast/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) labels
3. Comment on the issue saying you are going to work on it.
4. Get familiar with the developer guide relevant for you:
    * TBA
5. Code! Make sure to update unit tests!
6. When done, [create your pull request](https://github.com/nvidia/numbast/compare).
7. Verify that CI passes all [status checks](https://help.github.com/articles/about-status-checks/), or fix if needed.
8. Wait for other developers to review your code and update code as needed.
9. Once reviewed and approved, a numbast developer will merge your pull request.

Remember, if you are unsure about anything, don't hesitate to comment on issues and ask for clarifications!

## Releases

The release process for Numbast involves the following steps:

- Make sure your local main is top-of-tree.
- Open a PR to update `VERSION` in repo root to the desired version.
- Generate a short changelog with `git log v<PREVIOUS_VERSION>..HEAD --oneline --pretty=format:"- %s"`
- Put the changelog in the version update PR description.
- Once `main` is updated, tag the release:
```
git checkout main && git pull
git tag -a v<VERSION>
```
- For the tag annotation, paste the same changelog as above, like this:
```
v<VERSION>

- ... (bullet points on release items)
```
- Push the tag:
```
git push git@github.com:NVIDIA/numbast.git v<VERSION>
```

## Attribution
Portions adopted from https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md
