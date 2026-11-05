# Security

mercymek is a local research/demo application. `scripts/run_demo.py` binds to
`127.0.0.1` by default. It has no user authentication or multi-user isolation;
all connected clients share the active case and investigation state. Keep the
server on a trusted local machine. A public GitHub repository does not make the
server suitable for public internet hosting.

Browser WebSocket requests must come from the application's own origin. Native
local WebSocket clients may omit Origin. This is a browser-origin boundary,
not authentication. Case IDs and file paths are restricted to the case directory,
and speech URLs accept only generated hexadecimal cache identifiers.

Store credentials in the ignored `.env` file. Keep signing keys, uploaded clips,
API results, model weights, notebook outputs and local logs out of Git. The
Mapillary client token is intentionally sent to the browser when configured;
Meta, Hugging Face, Kaggle and ElevenLabs credentials stay on the server.

Before publishing, run `python scripts/check_public_repo.py --history`. Its
pattern and known-local-key scan reduces accidental disclosure; it cannot prove
that every possible secret is absent. Rotate a credential if it was disclosed;
removing it only from the current file does not remove it from Git history.

Do not post secrets, private footage or exploit details in public issues. Use
GitHub private vulnerability reporting when enabled; otherwise ask the maintainer
for a private reporting channel before sharing sensitive details.
