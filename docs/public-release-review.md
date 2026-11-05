# Public release review — 2026-11-05

Application version: **1.0.0**. The source and local delivery archives have been
prepared for public GitHub review. The public repository is
[helizac/mercymek](https://github.com/helizac/mercymek).

## Verified locally

- Full Windows/Python 3.11 environment with saved case assets: **191 tests passed**.
- Independent source-only checkout, newly created environment, hashed CPU
  dependencies, no API keys or case data: **175 passed, 4 skipped**. The skips
  require saved cases, a SAM smoke artifact or optional GLB exports.
- Generated all **18 vehicle GLBs** from the clean checkout; its reconstruction
  tests then passed (**10 tests**). Regenerated fitting points match the bundled
  source asset exactly, and a built wheel contains that asset.
- `pip check` in the new environment found no conflicting requirements.
- Public-source audit and all reachable Git history passed pattern and known-local
  credential checks. Both case-data archives also passed their credential check.
  These scans are limited detection checks, not proof that every possible secret
  can be detected.
- The packaged demo verification extracts source and both data archives, checks
  all seven original evidence seals, serves video/3D assets, and exercises cached
  commands, speech and plate enhancement with API credentials removed. Its
  machine-readable result is `deliverables/release-validation.json` and identifies
  the exact source ZIP by SHA-256.

## Corrections made

- Complete setup steps, cached/live requirements and separate source/data delivery
  instructions; historical design documents clearly marked as historical.
- Consistent package/API version, changelog and first-release/tag procedure.
- Hash-locked CPU runtime/test dependencies and a credential-free GitHub Actions
  workflow with pinned action revisions.
- Case-directory validation, speech-cache key validation and browser WebSocket
  origin validation, with regression coverage.
- Lightweight fitting geometry included in the source/package, removing a hidden
  dependency on ignored local model exports.
- Leaflet's original license and complete Nexar data terms retained; both data
  archives carry redistribution notices and credits.
- Ignore rules for local credentials, keys, environments, data, outputs and logs;
  source/history checking script for future releases.
- Remove the unused frontend prototype, two superseded revision-2 recording checks
  and the old implementation proposal; current recording and application code stay
  in the source package.

Project-authored code intentionally has **no added license**, as requested.
Third-party notices do not grant a license to that code.

## Publication and release assets

Source repository: [helizac/mercymek](https://github.com/helizac/mercymek).
The Linux workflow runs when source changes are pushed; current execution results
are available in [GitHub Actions](https://github.com/helizac/mercymek/actions).

Application version **1.0.0** is prepared locally. The release tag and GitHub Release
are separate publication steps: tag the passing release commit as `v1.0.0`, upload
the documented release assets, then add the actual download/demo URLs. Large data
archives, the film and local environments remain outside Git.

Local checks ran on Windows with Python 3.11. Optional live model/GPU reconstruction
paths are separate from the saved-demo checks. The application remains a local
research/demo server with shared state and no user authentication.

Follow [releasing.md](releasing.md) for publication.
