# Changelog

The application version is **1.0.0** in `pyproject.toml` and `mercy/__init__.py`.
This initial public release is prepared locally and has not yet been tagged.
Video edit revisions (v1–v6), model names and the historical iteration headings in
[CHANGES.md](CHANGES.md) are separate from application versions.

## 1.0.0 — initial public release, prepared

- Voice-controlled video, estimated 3D scenes, location, report and evidence views.
- Visual/OCR-based geolocation, plate enhancement, optical-flow replay and what-if analysis.
- Accident statements, signed evidence manifests and cached speech for saved cases.
- Separate source and sealed case-data archives; a three-minute English narrated demo.
- Validate named-vehicle commands and compound paperwork requests.
- Restrict case paths and speech-cache identifiers; reject foreign browser WebSocket origins.
- Public-source/history audit, consistent API/package versions, CI, hashed CPU dependency lock, setup instructions and third-party notices.
- Include complete Nexar redistribution terms in both source and data archives.
- Bundle the lightweight fitting geometry so clean checkouts do not depend on ignored local model exports.
- Remove the unused frontend prototype, superseded revision-2 recording checks and historical implementation proposal.

Release procedure: [docs/releasing.md](docs/releasing.md).
