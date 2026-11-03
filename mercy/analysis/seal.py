"""Evidence seal: a signed SHA-256 manifest of the source footage and every derived artifact.

Anyone with the public key can check that nothing in the case was changed after sealing (chain of custody). The
signing key is created once per installation in data/keys/ and never leaves the machine.
"""
from __future__ import annotations

import base64
import hashlib
import json
import pathlib
import time

from mercy import config

KEY = config.ROOT / "data" / "keys" / "seal_ed25519.pem"
SKIP = {"seal.json", "claims.json", "notes.json"}          # living files: change during the investigation
SKIP_DIRS = {"live", "frames", "frames4d", "masks", "geo"}  # bulky intermediates; their products are sealed


def _key():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    if KEY.exists():
        return serialization.load_pem_private_key(KEY.read_bytes(), password=None)
    KEY.parent.mkdir(parents=True, exist_ok=True)
    k = Ed25519PrivateKey.generate()
    KEY.write_bytes(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                    serialization.NoEncryption()))
    return k


def _pub_b64(k) -> str:
    from cryptography.hazmat.primitives import serialization
    return base64.b64encode(k.public_key().public_bytes(serialization.Encoding.Raw,
                                                        serialization.PublicFormat.Raw)).decode()


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def manifest(case: pathlib.Path) -> list[dict]:
    out = []
    for p in sorted(case.rglob("*")):
        rel = p.relative_to(case).as_posix()
        if p.is_file() and p.name not in SKIP and rel.split("/")[0] not in SKIP_DIRS:
            out.append({"path": rel, "sha256": sha256(p), "bytes": p.stat().st_size})
    return out


def root_of(files: list[dict]) -> str:
    return hashlib.sha256("".join(f"{f['path']}:{f['sha256']}\n" for f in files).encode()).hexdigest()


def seal(case: pathlib.Path) -> dict:
    files = manifest(case)
    root = root_of(files)
    k = _key()
    msg = f"mercymek-seal|{case.name}|{root}".encode()
    s = {"case": case.name, "created": time.time(), "algorithm": "SHA-256 manifest, Ed25519 signature",
         "root": root, "signature": base64.b64encode(k.sign(msg)).decode(), "public_key": _pub_b64(k),
         "source_sha256": next((f["sha256"] for f in files if f["path"] == "source_full.mp4"), None),
         "files": files}
    (case / "seal.json").write_text(json.dumps(s, indent=1))
    return s


def verify(case: pathlib.Path) -> dict:
    """Recompute every hash and check the signature; lists files changed, added or missing since sealing."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    p = case / "seal.json"
    if not p.exists():
        return {"sealed": False}
    s = json.loads(p.read_text())
    try:
        Ed25519PublicKey.from_public_bytes(base64.b64decode(s["public_key"])).verify(
            base64.b64decode(s["signature"]), f"mercymek-seal|{s['case']}|{s['root']}".encode())
        sig_ok = root_of(s["files"]) == s["root"]
    except Exception:
        sig_ok = False
    now = {f["path"]: f["sha256"] for f in manifest(case)}
    then = {f["path"]: f["sha256"] for f in s["files"]}
    changed = sorted(k for k in then if k in now and now[k] != then[k])
    missing = sorted(k for k in then if k not in now)
    added = sorted(k for k in now if k not in then)
    return {"sealed": True, "signature_valid": sig_ok, "intact": sig_ok and not changed and not missing,
            "files": len(then), "changed": changed, "missing": missing, "added": added,
            "root": s["root"], "created": s["created"]}
