// AudioWorklet: float32 mic samples (context runs at 24 kHz) -> int16 LE frames of 80 ms (1920 samples).
class Pcm24k extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Int16Array(1920); this.n = 0;
    this.port.onmessage = e => { if (e.data === 'flush' && this.n) { this.port.postMessage(this.buf.slice(0, this.n).buffer); this.n = 0; } };
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      const s = Math.max(-1, Math.min(1, ch[i]));
      this.buf[this.n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
      if (this.n === this.buf.length) { this.port.postMessage(this.buf.slice().buffer); this.n = 0; }
    }
    return true;
  }
}
registerProcessor('pcm24k', Pcm24k);
