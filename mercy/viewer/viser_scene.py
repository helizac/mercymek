"""viser 4D scene (section 8, step 5): background cloud, per-frame dynamic points, ego frustum, trajectories,
time slider and camera presets, driven by StateHub patches. Runs beside the HUD at :VISER_PORT."""
from __future__ import annotations

import json
import threading
import time

import numpy as np

from mercy.config import log


def _hex(c: str):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def to_viser(p):
    """Road frame (x right, y up, z forward) -> right-handed viewer frame (x, y, -z) with +y up."""
    p = np.asarray(p, np.float32)
    return p * np.array([1, 1, -1], np.float32)


class Scene4D:
    def __init__(self, port: int = 8080):
        import viser
        self.s = viser.ViserServer(port=port)
        self.s.scene.set_up_direction("+y")
        self.case = None
        self.k = 0
        self.slider = None
        self._orbit_stop = threading.Event()
        log.info("viser on http://localhost:%d", port)

    # ---------------- data
    def load_case(self, case):
        self.case = case
        d = case.dir
        meta = json.loads((d / "cloud.json").read_text())
        buf = (d / "cloud.bin").read_bytes()
        L = meta["layout"]

        def arr(name, dt, cols):
            o = L[name]
            a = np.frombuffer(buf, dtype=dt, count=o["bytes"] // np.dtype(dt).itemsize, offset=o["offset"])
            return a.reshape(-1, cols) if cols > 1 else a

        self.bg_xyz = to_viser(arr("bg_xyz", np.float32, 3))
        self.bg_rgb = arr("bg_rgb", np.uint8, 3)
        self.dyn_xyz = to_viser(arr("dyn_xyz", np.float32, 3))
        self.dyn_obj = arr("dyn_obj", np.uint8, 1)
        self.frames = meta["frames"]
        self.obj_keys = meta["objects"]
        colors = {o["id"]: o["color"] for o in case.objects}
        self.obj_rgb = np.array([_hex(colors.get(k, "#8fa3b8")) for k in self.obj_keys] or [(200, 200, 200)], np.uint8)
        self.tracks = case.tracks
        self.times = np.array(self.tracks["times"])
        self.s.scene.reset()
        self.s.gui.reset()
        self.s.scene.add_point_cloud("/background", self.bg_xyz, self.bg_rgb, point_size=0.09)
        for o in case.objects:
            pts = None
            if o["id"] == "ego":
                pts = np.array([e["pos"] for e in self.tracks["ego"]])
            elif o["id"] in self.tracks.get("tracks", {}):
                pts = np.array([r["center"] for r in self.tracks["tracks"][o["id"]]])
            if pts is not None and len(pts) >= 2:
                pts = pts.copy(); pts[:, 1] = 0.15
                h = self.s.scene.add_spline_catmull_rom(f"/traj/{o['id'].replace(':', '_')}", to_viser(pts),
                                                        color=_hex(o["color"]), line_width=4.0)
                h.visible = o["role"] != "other"
        T = len(self.frames)
        self.slider = self.s.gui.add_slider("Frame", min=0, max=max(T - 1, 1), step=1, initial_value=0)
        self.slider.on_update(lambda _: self.set_frame(int(self.slider.value)))
        for name in ("top", "side", "driver", "orbit"):
            b = self.s.gui.add_button(name.title())
            b.on_click(lambda _, n=name: self.camera_preset(n))
        self.set_frame(0)

    def dynamic_points(self, k):
        f = self.frames[k]
        sl = slice(f["start"], f["start"] + f["count"])
        return self.dyn_xyz[sl], self.obj_rgb[self.dyn_obj[sl]]

    def ego_pose(self, k):
        e = self.tracks["ego"][k]
        fwd = to_viser(e["fwd"]); up = to_viser(e["up"])
        z = fwd / np.linalg.norm(fwd); y = -up / np.linalg.norm(up); x = np.cross(y, z)
        R = np.stack([x, y, z], 1)                   # camera axes (OpenCV: x right, y down, z forward)
        from scipy.spatial.transform import Rotation
        q = Rotation.from_matrix(R).as_quat()        # x y z w
        return to_viser(e["pos"]), np.array([q[3], q[0], q[1], q[2]])

    def focus_point(self, k):
        fp = self.case.physics.get("impact", {}).get("point_xz")
        if fp:
            return to_viser([fp[0], 0.5, fp[1]])
        return self.ego_pose(k)[0]

    # ---------------- view
    def set_frame(self, k: int):
        if self.case is None:
            return
        k = int(np.clip(k, 0, len(self.frames) - 1))
        self.k = k
        pts, cols = self.dynamic_points(k)
        if len(pts):
            self.s.scene.add_point_cloud("/dynamic", pts, cols, point_size=0.14)
        pos, wxyz = self.ego_pose(k)
        self.s.scene.add_camera_frustum("/ego", fov=float(self.tracks.get("fov_y", 0.9)),
                                        aspect=float(self.tracks.get("aspect", 16 / 9)), scale=1.2,
                                        color=(127, 209, 199), wxyz=wxyz, position=pos)
        if self.slider is not None and int(self.slider.value) != k:
            self.slider.value = k

    def camera_preset(self, name: str):
        if self.case is None:
            return
        self._orbit_stop.set()
        target = self.focus_point(self.k)
        if name == "orbit":
            self._orbit_stop = threading.Event()
            threading.Thread(target=self._orbit, args=(target, self._orbit_stop), daemon=True).start()
            return
        if name == "driver":
            pos, _ = self.ego_pose(self.k)
            look = pos + to_viser(self.tracks["ego"][self.k]["fwd"]) * 10
            pos, target = pos, look
        else:
            pos = target + (np.array([0, 45, 0.01]) if name == "top" else np.array([22, 5, 0]))
        for c in self.s.get_clients().values():
            c.camera.position, c.camera.look_at = tuple(pos), tuple(target)

    def _orbit(self, target, stop):
        t0 = time.time()
        while not stop.is_set() and time.time() - t0 < 4.0:
            a = (time.time() - t0) / 4.0 * 2 * np.pi
            pos = target + np.array([np.cos(a) * 24, 10, np.sin(a) * 24])
            for c in self.s.get_clients().values():
                c.camera.position, c.camera.look_at = tuple(pos), tuple(target)
            time.sleep(1 / 30)

    # ---------------- hub listener
    def on_state(self, patch: dict):
        if self.case is None:
            return
        if "t" in patch:
            k = int(np.argmin(np.abs(self.times - float(patch["t"]))))
            if k != self.k:
                self.set_frame(k)
        if "camera" in patch and patch["camera"] in ("top", "side", "driver", "orbit"):
            self.camera_preset(patch["camera"])
