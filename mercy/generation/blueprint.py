"""Muse Image restyle of the deterministic diagram into a tactical blueprint (section 10.2).
Output is never evidence; the HUD watermarks it with CSS."""
from __future__ import annotations

import base64
import pathlib

from mercy import config
from mercy.config import log
from mercy.meta_client import meta_client

BLUEPRINT_PROMPT = (
    "Restyle this accident diagram as a futuristic forensic tactical blueprint: deep navy background, "
    "cyan grid lines, glowing neon vehicle outlines and trajectories, small HUD-style labels. "
    "Keep every trajectory, vehicle position, label and the impact marker exactly where they are. "
    "Do not add vehicles, people or text that is not in the original.")


def make_blueprint(diagram: pathlib.Path, out: pathlib.Path, timeout: float = 90) -> pathlib.Path:
    with open(diagram, "rb") as img:
        r = meta_client().with_options(timeout=timeout).images.edit(
            model=config.IMAGE_MODEL, prompt=BLUEPRINT_PROMPT, image=img, n=1,
            extra_body={"reasoning_strength": "low", "output_format": "webp",
                        "tool_enablement": {"enable_web_search": False, "enable_image_search": False,
                                            "enable_shell": False}})
    out.write_bytes(base64.b64decode(r.data[0].b64_json))
    log.info("blueprint -> %s (%d KB)", out.name, out.stat().st_size // 1024)
    return out


def run(case: pathlib.Path, force: bool = False) -> pathlib.Path:
    out = case / "blueprint.webp"
    if out.exists() and not force:
        return out
    return make_blueprint(case / "diagram.png", out)
