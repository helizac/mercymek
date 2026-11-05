# Public GitHub release

Application version: **1.0.0**. The intended first tag is **v1.0.0**. No release tag
or GitHub remote is created by the preparation scripts. Commit the reviewed work
before tagging; a tag must identify the commit containing the released version.
Film edit names such as `demo-v6.mp4` do not change the application version.

## Local verification

From an activated Python 3.11 environment in the source checkout:

```bash
python -m pip install --require-hashes -r requirements-dev.lock
python -m pip install --no-deps -e ".[dev]"
python scripts/check_public_repo.py --history --expected-version v1.0.0
python -m pytest -q -ra
python scripts/prepare_delivery.py package
python scripts/prepare_delivery.py verify
```

The final verification needs both local case-data archives. The public CI job
runs without those large data files or credentials; tests needing saved cases or
API smoke artifacts skip accordingly. The full local suite covers the saved data.

Inspect `git status` and `git diff` before staging. `.env`, `.env.*` (except the
empty example), signing keys, `data/`, `deliverables/`, environments and model
weights stay outside Git. Do not paste credentials into notebook cells or outputs.
The source audit scans current publishable files and, with `--history`, reachable
Git history. It reports locations, never secret values.

## Versioning and publication

1. Keep `pyproject.toml`, `mercy/__init__.py` and `CHANGELOG.md` consistent. Use
   MAJOR.MINOR.PATCH application versions; update the changelog for each release.
2. Create an empty public GitHub repository and copy its actual clone URL. The
   project-authored source intentionally has no license; keep the bundled
   third-party notices. Do not select an automatic MIT/Apache license on GitHub.
3. Commit all reviewed source changes, including new documentation, workflow and
   tests. Confirm the working tree contains no unintended additions.
4. Add that repository as `origin`, push the intended branch, and wait for CI.
5. Tag the passing release commit and push the tag:

   ```bash
   git tag -a v1.0.0 -m "mercymek 1.0.0"
   git push origin v1.0.0
   ```

6. Create a GitHub Release from that tag. Attach the assets listed in
   [delivery.md](delivery.md), including the two data ZIPs, final MP4, subtitles,
   `CREDITS.txt`, source ZIP and checksums. Case/media licenses and attribution
   must accompany the data; do not attach `.env`, private signing keys or the
   ignored production work directory.
7. Add the actual hosted release/demo links to README and the submission portal.
   Verify downloads from a fresh checkout before announcing the release.

The repository is a source-checkout application: keep `frontend/`, `scripts/` and
`mercy/` together. It is not published as a standalone PyPI package. The public
source does not contain the prepared case assets, model checkpoints or private
Kaggle runtime caches; owners rebuilding those caches need their own accounts.

The [CI workflow](../.github/workflows/ci.yml) checks the source, history and tag
version, then installs CPU dependencies and runs credential-free tests. Its first
actual GitHub run happens after the repository has been pushed.

## Refresh runtime/test dependencies

`requirements-dev.lock` is a universal, hashed CPU-runtime/test resolution. It is
separate from optional reconstruction or GPU worker environments. To intentionally
update it, install uv and regenerate, then repeat the clean checkout and CI tests:

```bash
uv pip compile pyproject.toml --extra dev --universal --torch-backend cpu --generate-hashes --emit-index-url --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match --upgrade --output-file requirements-dev.lock
```

Both registries in that command are public upstream registries. Do not generate
a committed lockfile containing authenticated registry URLs. Runtime and test
package hashes are checked during installation; build tooling and external model
weights are separate from this lock.
