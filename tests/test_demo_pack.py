"""A judge's extracted demo must verify against the original evidence seal."""
import zipfile

import pytest

from mercy.analysis import seal
from scripts import pack_demo


def test_pack_preserves_original_seal_after_clean_extraction(tmp_path, monkeypatch):
    source = tmp_path / "source"
    case = source / "data" / "cases" / "example"
    case.mkdir(parents=True)
    (case / "meta.json").write_text("{}")
    (case / "physics.json").write_text("{}")
    (case / "clip.mp4").write_bytes(b"source clip")
    (case / "recon.npz").write_bytes(b"sealed server-side reconstruction")
    (case / "environment").mkdir()
    (case / "environment" / "source.png").write_bytes(b"sealed reference")
    (source / ".env").write_text("SECRET=not-for-distribution")
    monkeypatch.setattr(seal, "KEY", source / "data" / "keys" / "seal.pem")
    original = seal.seal(case)
    monkeypatch.setattr(pack_demo, "ROOT", source)
    archive = tmp_path / "demo.zip"
    pack_demo.pack(["example"], archive)
    recipient = tmp_path / "recipient"
    monkeypatch.setattr(pack_demo, "ROOT", recipient)
    pack_demo.unpack(archive)
    received = recipient / "data" / "cases" / "example"
    result = seal.verify(received)
    assert result["intact"]
    assert result["root"] == original["root"]
    assert result["files"] == len(original["files"])
    with zipfile.ZipFile(archive) as z:
        assert not any("keys/" in n or n.endswith(".env") for n in z.namelist())
    (received / "recon.npz").write_bytes(b"tampered")
    assert not seal.verify(received)["intact"]


def test_pack_refuses_changed_evidence(tmp_path, monkeypatch):
    case = tmp_path / "data" / "cases" / "example"
    case.mkdir(parents=True)
    (case / "meta.json").write_text("{}")
    monkeypatch.setattr(seal, "KEY", tmp_path / "private.pem")
    seal.seal(case)
    (case / "meta.json").write_text('{"changed":true}')
    monkeypatch.setattr(pack_demo, "ROOT", tmp_path)
    archive = tmp_path / "demo.zip"
    with pytest.raises(ValueError, match="not intact"):
        pack_demo.pack(["example"], archive)
    assert not archive.exists()


@pytest.mark.parametrize("name", ["../README.md", "data/cases/../../.env", "C:/Windows/file", "data/keys/private.pem"])
def test_unpack_rejects_paths_outside_demo_data(tmp_path, monkeypatch, name):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(name, b"unexpected")
    monkeypatch.setattr(pack_demo, "ROOT", tmp_path / "recipient")
    with pytest.raises(ValueError, match="archive path"):
        pack_demo.unpack(archive)


def test_nexar_pack_keeps_complete_notice_separate_from_signed_evidence(tmp_path, monkeypatch):
    source = tmp_path / "source"
    case = source / "data/cases/example"
    case.mkdir(parents=True)
    (case / "meta.json").write_text('{"source":"Nexar"}')
    monkeypatch.setattr(pack_demo, "ROOT", source)
    archive = tmp_path / "demo.zip"
    with pytest.raises(ValueError, match="Nexar redistribution"):
        pack_demo.pack(["example"], archive)
    notice = source / "third_party/NEXAR_LICENSE.txt"
    notice.parent.mkdir()
    notice.write_text("Complete upstream copyright, terms and disclaimer")
    pack_demo.pack(["example"], archive)
    recipient = tmp_path / "recipient"
    monkeypatch.setattr(pack_demo, "ROOT", recipient)
    pack_demo.unpack(archive)
    assert (recipient / "data/licenses/NEXAR_LICENSE.txt").read_text() == notice.read_text()
