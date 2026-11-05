"""Reproducible 180-second submission film: ElevenLabs audio, captions and FFmpeg edit.

    python scripts/demo_video.py voices
    node scripts/record_demo.cjs
    python scripts/demo_video.py edit
    python scripts/demo_video.py verify

Media stays in ignored deliverables/. Credentials are read from .env and never
included in metadata. Existing speech is reused, so editing costs no API credits.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mercy import config  # loads the project .env

OUT = ROOT / "deliverables"
MEDIA = OUT / "production" / "revision6"
PLAN = json.loads((ROOT / "docs" / "demo-storyboard.json").read_text(encoding="utf-8"))


def binary(name):
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"Install {name} and add it to PATH")
    return found


def run(args):
    result = subprocess.run([str(x) for x in args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(result.stderr[-6000:] or result.stdout[-6000:])
    return result.stdout


def probe(path):
    return json.loads(run([binary("ffprobe"), "-v", "error", "-show_format", "-show_streams", "-of", "json", path]))


def voices():
    import httpx
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise RuntimeError("ELEVENLABS_API_KEY is required to create the narration")
    voice = os.environ.get("ELEVEN_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
    model = os.environ.get("DEMO_ELEVEN_MODEL", "eleven_v4")
    MEDIA.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=180, headers={"xi-api-key": key}) as client:
        for i, shot in enumerate(PLAN["shots"]):
            audio = MEDIA / f"{shot['id']}.mp3"
            timings = MEDIA / f"{shot['id']}.alignment.json"
            identity = hashlib.sha256(f"{voice}|{model}|{shot['narration']}".encode()).hexdigest()
            if audio.exists() and timings.exists() and json.loads(timings.read_text()).get("identity") == identity:
                print(f"cached {shot['id']}", flush=True)
                continue
            body = {"text": shot["narration"], "model_id": model,
                    "voice_settings": {"stability": 0.55, "similarity_boost": 0.8}, "seed": 20261104}
            if i:
                body["previous_text"] = PLAN["shots"][i - 1]["narration"]
            if i + 1 < len(PLAN["shots"]):
                body["next_text"] = PLAN["shots"][i + 1]["narration"]
            response = client.post(f"https://api.elevenlabs.io/v1/text-to-speech/{voice}/with-timestamps",
                                   params={"output_format": "mp3_44100_128"}, json=body)
            response.raise_for_status()
            result = response.json()
            audio.write_bytes(base64.b64decode(result.pop("audio_base64")))
            result.update(identity=identity, voice=voice, model=model, provider="ElevenLabs", text=shot["narration"],
                          request_id=response.headers.get("request-id"), characters=response.headers.get("character-cost"))
            timings.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"generated {shot['id']}: {float(probe(audio)['format']['duration']):.2f}s", flush=True)


def stamp(seconds, ass=False):
    units = 100 if ass else 1000
    ticks = round(seconds * units)
    h, ticks = divmod(ticks, 3600 * units)
    m, ticks = divmod(ticks, 60 * units)
    s, frac = divmod(ticks, units)
    return f"{h}:{m:02}:{s:02}.{frac:02}" if ass else f"{h:02}:{m:02}:{s:02},{frac:03}"



def captions(shot, start, delay=.25):
    """Sentence-aware, nonoverlapping captions at the original speech speed."""
    data = json.loads((MEDIA / f"{shot['id']}.alignment.json").read_text())
    a = data.get("normalized_alignment") or data["alignment"]
    text = "".join(a["characters"])
    words = list(re.finditer(r"\S+", text))
    sentences, sentence = [], []
    weak = {"a", "an", "the", "and", "or", "of", "to", "with", "from", "in", "on", "for", "their", "its"}
    for i, word in enumerate(words):
        sentence.append(word)
        if re.search(r"[.!?]$", word.group()) or i==len(words)-1:
            sentences.append(sentence);sentence=[]
    groups=[]
    for sentence in sentences:
        remaining=sentence
        while len(" ".join(w.group() for w in remaining))>82:
            total=len(" ".join(w.group() for w in remaining))
            target=total/math.ceil(total/78)
            candidates=[]
            for j in range(2,len(remaining)-1):
                left=" ".join(w.group() for w in remaining[:j])
                right=" ".join(w.group() for w in remaining[j:])
                if len(left)>82 or len(left)<22 or len(right)<22:continue
                score=abs(len(left)-target)
                if left.endswith((",",";",":")):score-=12
                if remaining[j-1].group().lower() in weak:score+=18
                numeric=re.match(r"^(\d|zero|one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)",remaining[j-1].group().lower())
                if numeric and remaining[j].group().lower() in {"to","percent.","percent","metres","kilometres","seconds"}:
                    score+=50
                candidates.append((score,j))
            if not candidates:break
            j=min(candidates)[1];groups.append(remaining[:j]);remaining=remaining[j:]
        groups.append(remaining)
    result=[]
    for i, group in enumerate(groups):
        lo = start+delay+a["character_start_times_seconds"][group[0].start()]
        hi = start+delay+a["character_end_times_seconds"][group[-1].end()-1]+.1
        if i+1<len(groups):
            following=start+delay+a["character_start_times_seconds"][groups[i+1][0].start()]
            hi=min(hi,following-.025)
        hi=min(hi,start+shot["duration"]-.08)
        line=" ".join(w.group() for w in group)
        for spoken, written in [("L M fifty-eight B L Z", "LM58 BLZ"),
                                ("A ten", "A10"), ("three D", "3D"),
                                ("M twenty-five", "M25")]:
            line=line.replace(spoken,written)
        if len(line)>43:
            options=[]
            tokens=line.split()
            for j in range(1,len(tokens)):
                left=" ".join(tokens[:j]);right=" ".join(tokens[j:])
                if max(len(left),len(right))>43:continue
                score=abs(len(left)-len(right))+(15 if tokens[j-1].lower() in weak else 0)
                options.append((score,left+"\n"+right))
            line=min(options)[1] if options else "\n".join(textwrap.wrap(line,width=43,break_long_words=False,break_on_hyphens=False))
        result.append((lo,hi,line))
    return result


def edit(refresh=None):
    """Explicit source intervals, original voice speed and clean picture cuts."""
    ffmpeg=binary("ffmpeg")
    records=json.loads((MEDIA/"recordings.json").read_text())
    work=MEDIA/"edit";work.mkdir(parents=True,exist_ok=True)
    subtitles=[];chapters=[];segments=[];start=0
    for shot in PLAN["shots"]:
        sid=shot["id"];duration=shot["duration"]
        audio=MEDIA/f"{sid}.mp3"
        spoken=float(probe(audio)["format"]["duration"])
        if spoken+.25>duration:
            raise ValueError(f"{sid}: natural speech {spoken:.2f}s does not fit {duration}s; revise the slot or script")
        subtitles.extend(captions(shot,start))
        chapters.append({"start":start,"end":start+duration,"title":shot["chapter"]})
        if refresh is not None and sid not in refresh:
            target=work/f"{sid}.mp4"
            if not target.exists():
                raise ValueError(f"No existing picture edit for {sid}; run a full edit first")
            segments.append(target);start+=duration
            continue
        record=records[sid];pictures=[]
        if abs(sum(c["duration"] for c in record["clips"])-duration)>.02:
            raise ValueError(f"{sid}: picture edit does not match the intended duration")
        for j,clip in enumerate(record["clips"]):
            piece=work/f"{sid}-{j}.mp4"
            source=MEDIA/clip["file"]
            args=["-loop","1","-i",source] if clip.get("image") else ["-ss",clip["in"],"-i",source]
            # Screen captures already have their final pixel size; no camera wobble,
            # invented frames, speed changes or hidden repeated approach replays.
            vf="fps=30,setsar=1,scale=1920:960:flags=lanczos,pad=1920:1080:0:0:color=0x07090b"
            if clip.get("image"):
                vf=("fps=30,setsar=1,scale=1920:1080:flags=lanczos" if clip.get("full_frame") else
                    "fps=30,setsar=1,scale=1840:850:force_original_aspect_ratio=decrease:flags=lanczos,pad=1920:1080:(ow-iw)/2:70:color=0x07090b")
            run([ffmpeg,"-y","-v","error",*args,"-an","-vf",vf,"-t",clip["duration"],
                 "-c:v","libx264","-preset","fast","-crf","18","-pix_fmt","yuv420p","-threads","4",piece])
            actual=float(probe(piece)["format"]["duration"])
            if abs(actual-clip["duration"])>.075:
                raise ValueError(f"Insufficient recorded frames for {sid}/{j}: {actual:.3f}s")
            pictures.append(piece)
        concat=work/f"{sid}.concat.txt"
        concat.write_text("".join(f"file '{p.as_posix()}'\n" for p in pictures),encoding="utf-8")
        target=work/f"{sid}.mp4"
        run([ffmpeg,"-y","-v","error","-f","concat","-safe","0","-i",concat,"-i",audio,
             "-map","0:v","-map","1:a","-c:v","copy","-af",f"adelay=250:all=1,apad,atrim=duration={duration}",
             "-c:a","aac","-b:a","192k","-ar","48000","-ac","2","-t",duration,target])
        segments.append(target)
        print(f"v6 edited {sid}: {start}-{start+duration}s; original voice speed",flush=True)
        start+=duration
    assert start==180
    for previous,current in zip(subtitles,subtitles[1:]):
        assert previous[1]<=current[0],f"Overlapping captions: {previous} / {current}"
    for lo,hi,text in subtitles:
        assert 0<=lo<hi<=180 and "mercy-mek" not in text.lower()
    srt="\n\n".join(f"{i}\n{stamp(a)} --> {stamp(b)}\n{t}" for i,(a,b,t) in enumerate(subtitles,1))+"\n"
    (OUT/"mercymek-demo.en.srt").write_text(srt,encoding="utf-8")
    ass="""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,Segoe UI,29,&H00F2F5F3,&H000000FF,&H000B0D0D,&H000B0D0D,0,0,0,0,100,100,0,0,1,1,0,2,180,180,28,1
Style: Label,Segoe UI,25,&H00C7D17F,&H000000FF,&H000B0D0D,&H000B0D0D,0,0,0,0,100,100,0,0,1,1,0,8,100,100,22,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    for a,b,text in subtitles:
        text=text.replace("\n",r"\N").replace("{","").replace("}","")
        ass+=f"Dialogue: 0,{stamp(a,True)},{stamp(b,True)},Caption,,0,0,0,,{text}\n"
    (work/"captions.ass").write_text(ass,encoding="utf-8-sig")
    (work/"concat.txt").write_text("".join(f"file '{p.as_posix()}'\n" for p in segments),encoding="utf-8")
    run([ffmpeg,"-y","-v","error","-f","concat","-safe","0","-i",work/"concat.txt","-c","copy",work/"picture.mp4"])
    metadata=";FFMETADATA1\ntitle=mercymek - Your voice runs the investigation\nartist=mercymek\ncomment=Revision 6. Actual application recordings. ElevenLabs narration at its original speed.\n"
    for c in chapters:
        metadata+=f"\n[CHAPTER]\nTIMEBASE=1/1000\nSTART={c['start']*1000}\nEND={c['end']*1000}\ntitle={c['title']}\n"
    (work/"chapters.txt").write_text(metadata,encoding="utf-8")
    old=Path.cwd();os.chdir(work)
    try:
        run([ffmpeg,"-y","-v","error","-i","picture.mp4","-i","chapters.txt","-i",OUT/"mercymek-demo.en.srt",
             "-map","0:v","-map","0:a","-map","2:s","-map_metadata","1","-map_chapters","1",
             "-vf","ass=captions.ass,fade=t=out:st=179.5:d=0.5","-af","loudnorm=I=-16:TP=-1.5:LRA=9,afade=t=out:st=179.85:d=0.15",
             "-c:v","libx264","-preset","slow","-crf","18","-pix_fmt","yuv420p","-r","30",
             "-c:a","aac","-b:a","192k","-ar","48000","-c:s","mov_text","-metadata:s:s:0","language=eng",
             "-disposition:s:0","0","-t","180","-movflags","+faststart","-threads","6",OUT/"mercymek-3-minute-demo-v6.mp4"])
    finally:
        os.chdir(old)
    shutil.copy2(OUT/"mercymek-3-minute-demo-v6.mp4",OUT/"mercymek-3-minute-demo.mp4")
    run([ffmpeg,"-y","-v","error","-i",OUT/"mercymek-3-minute-demo.mp4","-vn","-c:a","libmp3lame","-q:a","2",OUT/"mercymek-narration.mp3"])
    (OUT/"chapters.txt").write_text("\n".join(f"{c['start']//60}:{c['start']%60:02d} {c['title']}" for c in chapters)+"\n",encoding="utf-8")
    (OUT/"narration.txt").write_text("\n\n".join(s["narration"] for s in PLAN["shots"])+"\n",encoding="utf-8")
    verify()


def verify():
    movie = OUT / "mercymek-3-minute-demo.mp4"
    result = probe(movie)
    video = next(s for s in result["streams"] if s["codec_type"] == "video")
    audio = next(s for s in result["streams"] if s["codec_type"] == "audio")
    assert abs(float(result["format"]["duration"]) - 180) < .15
    assert (video["width"], video["height"]) == (1920, 1080)
    assert video["codec_name"] == "h264" and video["pix_fmt"] == "yuv420p"
    assert video["avg_frame_rate"] == "30/1" and audio["codec_name"] == "aac"
    assert any(s["codec_type"] == "subtitle" for s in result["streams"])
    run([binary("ffmpeg"), "-v", "error", "-i", movie, "-f", "null", "-"])
    report = {"file": movie.name, "seconds": float(result["format"]["duration"]), "resolution": "1920x1080", "fps": 30,
              "revision": 6, "speech_speed": "original", "caption_overlap": False,
              "video": "H.264 / yuv420p", "audio": "AAC / 48 kHz", "captions": "English, burned in + MP4 subtitle track + SRT",
              "bytes": movie.stat().st_size, "sha256": hashlib.sha256(movie.read_bytes()).hexdigest(), "full_decode": "passed"}
    (OUT / "video-validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["voices", "edit", "verify"])
    ap.add_argument("--refresh",nargs="+",choices=[s["id"] for s in PLAN["shots"]],help="Rebuild only these picture sections before the final composite")
    args=ap.parse_args()
    if args.action=="edit":
        edit(args.refresh)
    else:
        globals()[args.action]()
