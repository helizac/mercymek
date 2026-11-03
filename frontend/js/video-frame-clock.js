/* Keep comparison pixels and reconstruction time from the same presented video frame. */
export class VideoFrameClock {
  constructor(video) {
    this.video = video;
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d');
    this.time = null;
    this.generation = 0;
    this.callbacks = 0;
    this.revision = 0;
    video.addEventListener('seeked', () => {
      // Paused seeks are decoded without consistently producing rVFC in Chromium.
      // The evidence player uses the ingest's constant-frame-rate redacted copy.
      const generation=this.generation,target=video.currentTime;
      requestAnimationFrame(()=>requestAnimationFrame(()=>{
        if(generation!==this.generation || !video.paused || video.seeking || Math.abs(video.currentTime-target)>1e-6 || !this.fps)return;
        const time=Math.floor(target*this.fps+1e-5)/this.fps;
        if(this.time==null || Math.abs(this.time-time)>1e-6)this.capture(time,'decoded-cfr-seek');
      }));
    });
    if (!video.requestVideoFrameCallback) {
      for (const event of ['loadeddata', 'seeked', 'timeupdate'])
        video.addEventListener(event, () => this.capture(video.currentTime));
    }
  }

  reset(fps) {
    this.fps = fps;
    this.generation++;
    if (this.callback != null) this.video.cancelVideoFrameCallback?.(this.callback);
    this.time = null;
    const generation = this.generation;
    const tick = (_, metadata) => {
      if (generation !== this.generation) return;
      this.callbacks++;
      const sought=Math.floor(this.video.currentTime*this.fps+1e-5)/this.fps;
      if(!this.video.seeking && (!this.video.paused || !this.fps || Math.abs(metadata.mediaTime-sought)<.5/this.fps))
        this.capture(metadata.mediaTime,'presentation-timestamp');
      this.callback = this.video.requestVideoFrameCallback(tick);
    };
    if (this.video.requestVideoFrameCallback) this.callback = this.video.requestVideoFrameCallback(tick);
  }

  capture(time, source='media-clock-fallback') {
    const v = this.video;
    if (v.readyState < 2 || !v.videoWidth) return;
    if(this.time!=null && Math.abs(this.time-time)<1e-6)return;
    if (this.canvas.width !== v.videoWidth || this.canvas.height !== v.videoHeight) {
      this.canvas.width = v.videoWidth; this.canvas.height = v.videoHeight;
    }
    this.ctx.drawImage(v, 0, 0);
    this.time = time;
    this.source = source;
    this.revision++;
  }

  snapshot() {
    return this.time == null ? null : { image: this.canvas, time: this.time, revision: this.revision };
  }
}
