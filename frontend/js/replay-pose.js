// Shared interpolation: preserve body direction across +/-180 and tolerate missing depth observations.
export function keyAt(keys, t, { clampBefore = false } = {}) {
  if (!keys?.length) return null;
  if (t < keys[0][0]) return clampBefore ? keys[0].slice(1) : null;
  if (keys.length === 1 || t >= keys.at(-1)[0]) return keys.at(-1).slice(1);
  let lo = 0, hi = keys.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (keys[mid][0] > t) hi = mid; else lo = mid; }
  const a = keys[lo], b = keys[hi], f = Math.max(0, Math.min(1, (t-a[0]) / Math.max(b[0]-a[0], 1e-6)));
  const yawDelta = ((b[3]-a[3]+180)%360+360)%360-180;
  return [a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f, a[3]+yawDelta*f];
}

// Road +Z is forward; browser -Z is forward. Positive yaw means nose to road/image right.
// Reflect positions and direction vectors together; never flip only the body's left/right angle.
export function browserForward(yawDegrees) {
  const a = yawDegrees * Math.PI / 180;
  return [Math.sin(a), 0, -Math.cos(a)];
}

export function actorVisible(vehicle, t) {
  if (vehicle.model && vehicle.persistent) {
    return !vehicle.existence_range || (t >= vehicle.existence_range[0] && t <= vehicle.existence_range[1]);
  }
  return !vehicle.visible_range || (t >= vehicle.visible_range[0] && t <= vehicle.visible_range[1]);
}
