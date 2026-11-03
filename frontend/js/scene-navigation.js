/* User-controlled camera. No timer-driven orbit; moving the camera leaves follow mode. */
import * as THREE from 'three';

const UP = new THREE.Vector3(0, 1, 0);
const KEYS = new Set(['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE']);

export class SceneNavigation {
  constructor(canvas, camera, onManual) {
    this.canvas = canvas; this.camera = camera; this.onManual = onManual;
    this.keys = new Set(); this.pointer = null; this.lastTime = null; this.fast = false;
    canvas.tabIndex = 0; canvas.style.touchAction = 'none';
    canvas.setAttribute('aria-label', '3D environment: drag to look, wheel or W/S to move, A/D to strafe, Q/E to rise or descend');
    canvas.addEventListener('contextmenu', e => e.preventDefault());
    canvas.addEventListener('pointerdown', e => {
      if (e.button !== 0 && e.button !== 2) return;
      canvas.focus({ preventScroll: true }); canvas.setPointerCapture(e.pointerId);
      this.pointer = { id: e.pointerId, x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey };
      e.preventDefault();
    });
    canvas.addEventListener('pointermove', e => {
      const p = this.pointer; if (!p || p.id !== e.pointerId) return;
      const dx = e.clientX - p.x, dy = e.clientY - p.y; p.x = e.clientX; p.y = e.clientY;
      if (!dx && !dy) return;
      this.onManual();
      if (p.pan) {
        const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
        camera.position.addScaledVector(right, -dx * .025).addScaledVector(UP, dy * .025);
      } else {
        const a = new THREE.Euler().setFromQuaternion(camera.quaternion, 'YXZ');
        a.y -= dx * .004; a.x = THREE.MathUtils.clamp(a.x - dy * .004, -Math.PI * .48, Math.PI * .48);
        a.z = 0; camera.quaternion.setFromEuler(a);
      }
      camera.position.y = Math.max(.15, camera.position.y);
    });
    const up = e => { if (this.pointer?.id === e.pointerId) this.pointer = null; };
    canvas.addEventListener('pointerup', up); canvas.addEventListener('pointercancel', up);
    canvas.addEventListener('wheel', e => {
      e.preventDefault(); this.onManual();
      const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? canvas.clientHeight : 1;
      const forward = camera.getWorldDirection(new THREE.Vector3());
      camera.position.addScaledVector(forward, THREE.MathUtils.clamp(-e.deltaY * unit * .025, -20, 20));
      camera.position.y = Math.max(.15, camera.position.y);
    }, { passive: false });
    canvas.addEventListener('keydown', e => {
      if (!KEYS.has(e.code)) return;
      e.preventDefault(); e.stopPropagation(); this.onManual();
      this.keys.add(e.code); this.fast = e.shiftKey;
    });
    canvas.addEventListener('keyup', e => { this.keys.delete(e.code); this.fast = e.shiftKey; });
    canvas.addEventListener('blur', () => { this.keys.clear(); this.pointer = null; });
    window.addEventListener('blur', () => this.keys.clear());
  }

  update(now, enabled) {
    const dt = this.lastTime === null ? 0 : Math.min(.05, Math.max(0, (now - this.lastTime) / 1000));
    this.lastTime = now;
    if (!enabled) { this.keys.clear(); return; }
    if (!this.keys.size) return;
    const forward = this.camera.getWorldDirection(new THREE.Vector3());
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(this.camera.quaternion);
    const axis = (a, b) => Number(this.keys.has(a)) - Number(this.keys.has(b));
    const delta = forward.multiplyScalar(axis('KeyW', 'KeyS')).addScaledVector(right, axis('KeyD', 'KeyA'))
      .addScaledVector(UP, axis('KeyE', 'KeyQ'));
    if (delta.lengthSq()) this.camera.position.addScaledVector(delta.normalize(), dt * (this.fast ? 22 : 6));
    this.camera.position.y = Math.max(.15, this.camera.position.y);
  }
}
