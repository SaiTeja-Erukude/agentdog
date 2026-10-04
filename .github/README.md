# GitHub Actions

- **CI:** branch pushes and pull requests run tests, build and validate
  distributions, smoke-test the installed wheel, and upload artifacts.
- **Publish:** `agentdog-v*` tags or manual dispatch run the same checks before
  publishing to PyPI. Both workflows use Python 3.13.

Create a GitHub environment named `pypi` and register this
[PyPI Trusted Publisher](https://docs.pypi.org/trusted-publishers/using-a-publisher/):

| Field | Value |
| --- | --- |
| Owner | `SaiTeja-Erukude` |
| Repository | `agentdog` |
| Workflow | `publish.yml` |
| Environment | `pypi` |

No API token secret is needed. Update the version in `pyproject.toml` and
`agentdog/__init__.py`, commit, then push a matching tag:

```bash
git tag agentdog-v0.1.1
git push origin agentdog-v0.1.1
```

Manual dispatch publishes the selected ref without requiring a tag.
