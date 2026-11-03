/* Transport: one WebSocket to the backend (/ws). JSON frames {type, data}; binary frames = mic PCM.
   Push-to-talk: the HUD captures the mic, converts to 24 kHz int16 mono and streams it; the server relays it to
   Muse Voice Transcribe. The API key never reaches the browser. With MIC_SOURCE=server the server records instead. */
import { Emitter } from './util.js';

export class Net extends Emitter {
  constructor() { super(); this.q = []; this.ws = null; this.mic = null; this.micSource = 'browser'; }

  start() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = this.ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => { this.emit('conn', 'open'); this.q.splice(0).forEach(m => ws.send(m)); };
    ws.onclose = () => { this.emit('conn', 'closed'); setTimeout(() => this.start(), 1500); };
    ws.onmessage = ev => {
      if (typeof ev.data !== 'string') return;
      try { const m = JSON.parse(ev.data); this.emit(m.type, 'data' in m ? m.data : m); } catch (e) { console.warn('bad frame', e); }
    };
  }

  send(msg) {
    const s = JSON.stringify(msg);
    this.ws && this.ws.readyState === 1 ? this.ws.send(s) : this.q.push(s);
  }

  async pttDown() {
    this.send({ type: 'ptt', down: true });
    if (this.micSource !== 'browser') return true;
    try {
      if (!this.mic) {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
        const ctx = new AudioContext({ sampleRate: 24000 });
        await ctx.audioWorklet.addModule('js/pcm-worklet.js');
        const src = ctx.createMediaStreamSource(stream);
        const node = new AudioWorkletNode(ctx, 'pcm24k');
        node.port.onmessage = e => { if (this.mic.live && this.ws?.readyState === 1) this.ws.send(e.data); };
        src.connect(node);
        this.mic = { ctx, node, stream, live: false };
      }
      if (this.mic.ctx.state === 'suspended') await this.mic.ctx.resume();
      this.mic.live = true;
      return true;
    } catch (e) {
      this.emit('error', { message: 'Microphone unavailable: ' + e.message + '. Type the command instead.' });
      return false;
    }
  }

  pttUp() {
    if (this.mic) { this.mic.live = false; this.mic.node.port.postMessage('flush'); }
    setTimeout(() => this.send({ type: 'ptt', down: false }), 120);    // let the last 80 ms frame arrive first
  }
}
