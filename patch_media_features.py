"""
Deep OQ media upgrade. Run in the same folder as deep_oq.py:

    python patch_media_features.py

Adds:
  1. Image: 10 styles, 4 shapes (square, portrait, story, wide) and 1/2/4 pictures at once (Lab > Image)
  2. Photo: "Remove background" button (AI runs in the browser, first use downloads a small model)
  3. Video: "Make captions" (Whisper speech to text), editable, download as .srt,
     and burn them into the MP4 (falls back to the .srt if the server has no subtitle fonts)

Safe: every change must match exactly once, the result is syntax-checked, and a backup
deep_oq.py.bak_media is saved before anything is written.
"""
import shutil
import sys

FILE = sys.argv[1] if len(sys.argv) > 1 else "deep_oq.py"
EDITS = []


def edit(name, old, new):
    EDITS.append((name, old, new))


# ----------------------------------------------------------------- 1. IMAGE STYLES, SHAPES, VARIATIONS (server)
edit("image helpers", r'''def _pollinations_gen_url(prompt, seed):''', r'''IMG_STYLES = {
    "photo": "",
    "anime": "anime style, clean line art, vivid colours",
    "3d": "3D render, smooth materials, soft studio lighting, high detail",
    "watercolor": "watercolor painting, soft washes, visible paper texture",
    "oil": "oil painting, visible brush strokes, rich colours",
    "sketch": "pencil sketch, hand drawn, shaded lines on white paper",
    "cartoon": "cartoon illustration, bold outlines, flat bright colours",
    "logo": "flat vector logo design, simple shapes, clean, plain white background",
    "pixel": "pixel art, 16-bit game style",
    "cinematic": "cinematic film still, dramatic lighting, shallow depth of field, colour graded",
}
IMG_SIZES = {"square": (1024, 1024), "portrait": (896, 1120), "story": (720, 1280), "wide": (1280, 720)}


def _style_note(d):
    s = IMG_STYLES.get(str(d.get("style", "photo")))
    return ("\nSTYLE CHOSEN BY THE USER (this overrides the photo rule above, do not make it a photograph): " + s) if s else ""


def _pollinations_gen_url(prompt, seed, w=1024, h=1024):''')

edit("gen url size", r'''+ "&width=1024&height=1024&nologo=true&seed=" + str(seed))''',
     r'''+ "&width=" + str(w) + "&height=" + str(h) + "&nologo=true&seed=" + str(seed))''')
edit("free url def", r'''def _pollinations_url(prompt, seed):''', r'''def _pollinations_url(prompt, seed, w=1024, h=1024):''')
edit("free url size", r'''+ "?width=1024&height=1024&nologo=true&seed=" + str(seed))''',
     r'''+ "?width=" + str(w) + "&height=" + str(h) + "&nologo=true&seed=" + str(seed))''')

edit("style in prompt", r'''keep those words exactly. Output only the prompt:\n" + text)''',
     r'''keep those words exactly. Output only the prompt:\n" + text + _style_note(d))''')

edit("image block", r'''        seed = random.randint(1, 10**6)
        if POLLINATIONS_KEY:  # own key: the server fetches the picture (no logo, key stays secret)
            url = "/pimg?seed=" + str(seed) + "&p=" + urllib.parse.quote(p[:700])
        else:
            url = _pollinations_url(p, seed)
        return {"image": url, "answer": "Prompt used: " + p}''', r'''        w, h = IMG_SIZES.get(str(d.get("size", "square")), (1024, 1024))
        try:
            count = max(1, min(4, int(d.get("n") or 1)))
        except (TypeError, ValueError):
            count = 1
        urls = []
        for _ in range(count):
            seed = random.randint(1, 10**6)
            if POLLINATIONS_KEY:  # own key: the server fetches the picture (no logo, key stays secret)
                urls.append("/pimg?seed=" + str(seed) + "&w=" + str(w) + "&h=" + str(h)
                            + "&p=" + urllib.parse.quote(p[:700]))
            else:
                urls.append(_pollinations_url(p, seed, w, h))
        return {"image": urls[0], "images": urls, "answer": "Prompt used: " + p}''')

edit("pimg size", r'''    seed = re.sub(r"\D", "", request.args.get("seed", ""))[:9] or "1"''',
     r'''    seed = re.sub(r"\D", "", request.args.get("seed", ""))[:9] or "1"
    try:
        W = max(256, min(1536, int(request.args.get("w", "1024"))))
        H = max(256, min(1536, int(request.args.get("h", "1024"))))
    except ValueError:
        W = H = 1024''')
edit("pimg keyed", r'''_pollinations_gen_url(prompt, seed), {"Authorization"''', r'''_pollinations_gen_url(prompt, seed, W, H), {"Authorization"''')
edit("pimg free", r'''(_pollinations_url(prompt, seed), {}, "free")''', r'''(_pollinations_url(prompt, seed, W, H), {}, "free")''')

# ----------------------------------------------------------------- 1. IMAGE STYLES, SHAPES, VARIATIONS (page)
edit("lab options html", r'''<label class="lab" id="lfileW" hidden>''', r'''<div class="row2" id="limgopt" hidden>
          <label class="lab">Style<select class="in" id="lstyle"><option value="photo">Real photo</option><option value="anime">Anime</option><option value="3d">3D render</option><option value="watercolor">Watercolor</option><option value="oil">Oil painting</option><option value="sketch">Pencil sketch</option><option value="cartoon">Cartoon</option><option value="logo">Logo</option><option value="pixel">Pixel art</option><option value="cinematic">Cinematic</option></select></label>
          <label class="lab">Shape<select class="in" id="lsize"><option value="square">Square</option><option value="portrait">Portrait 4:5</option><option value="story">Story 9:16</option><option value="wide">Wide 16:9</option></select></label>
          <label class="lab">How many<select class="in" id="ln"><option value="1">1 picture</option><option value="2">2 pictures</option><option value="4">4 pictures</option></select></label>
        </div>
        <label class="lab" id="lfileW" hidden>''')
edit("lab options show", r'''$('#lplaceW').hidden=!w;$('#lareaW').hidden=!w;''',
     r'''$('#lplaceW').hidden=!w;$('#lareaW').hidden=!w;$('#limgopt').hidden=lmod!=='image';''')
edit("lab send options", r'''area:$('#larea').value,image:(lmod==='vision'||lmod==='agri')?limg:''});''',
     r'''area:$('#larea').value,style:$('#lstyle').value,size:$('#lsize').value,n:$('#ln').value,image:(lmod==='vision'||lmod==='agri')?limg:''});''')
edit("lab show pictures", r'''o.innerHTML=(j.image?'<img id="limgout" alt="Generated picture" style="max-width:100%;border-radius:12px;margin-bottom:12px">':'')''',
     r'''o.innerHTML=((j.images||(j.image?[j.image]:[])).map(u=>'<a href="'+esc(u)+'" target="_blank" rel="noopener"><img class="limgout" data-s="'+esc(u)+'" alt="Generated picture" style="max-width:100%;border-radius:12px;margin:0 0 12px"></a>').join(''))''')
edit("lab load pictures", r'''if(j.image){const im=$('#limgout');im.onerror=()=>{im.remove();lstatus('The picture service did not answer. Try again in a moment.')};im.src=j.image}''',
     r'''if(j.image||j.images){document.querySelectorAll('.limgout').forEach(im=>{im.onerror=()=>{im.remove();lstatus('A picture did not load. Try again in a moment.')};im.src=im.dataset.s})}''')

# ----------------------------------------------------------------- 2. REMOVE BACKGROUND (browser)
edit("rmbg button", r'''<button type="button" id="p-auto">Auto enhance</button>''',
     r'''<button type="button" id="p-auto">Auto enhance</button><button type="button" id="p-rmbg">Remove background</button>''')
edit("rmbg code", r'''$('#p-r180').onclick=()=>{PH.rot=(PH.rot+2)%4;PH.rc=null;queue()};''', r'''$('#p-rmbg').onclick=async()=>{
  if(!PH.src)return;
  pai('Removing the background. The first time it downloads a small AI model, so wait up to a minute.');$('#p-rmbg').disabled=true;
  try{
    const mod=await import('https://cdn.jsdelivr.net/npm/@imgly/background-removal@1.5.5/+esm');
    const k=Math.min(1,1600/Math.max(PH.src.width,PH.src.height)),c=document.createElement('canvas');
    c.width=Math.round(PH.src.width*k);c.height=Math.round(PH.src.height*k);
    c.getContext('2d').drawImage(PH.src,0,0,c.width,c.height);
    const blob=await new Promise(r=>c.toBlob(r,'image/png'));
    const out=await (mod.removeBackground||mod.default)(blob);
    const url=URL.createObjectURL(out),img=new Image();
    await new Promise((ok,no)=>{img.onload=ok;img.onerror=()=>no(new Error('bad'));img.src=url});
    const s=document.createElement('canvas');s.width=img.naturalWidth;s.height=img.naturalHeight;
    s.getContext('2d').drawImage(img,0,0);URL.revokeObjectURL(url);
    PH.src=s;PH.rc=null;$('#p-fmt').value='image/png';queue();
    pai('Background removed. Save as PNG to keep it see-through.');
  }catch(err){pai('Could not remove the background. Check your internet connection and try a smaller photo.')}
  $('#p-rmbg').disabled=false;
};
$('#p-r180').onclick=()=>{PH.rot=(PH.rot+2)%4;PH.rc=null;queue()};''')

# ----------------------------------------------------------------- 3. VIDEO CAPTIONS (server)
edit("video skipped flag", r'''work = tempfile.mkdtemp(prefix="oq_")''', r'''skipped = False
    work = tempfile.mkdtemp(prefix="oq_")''')
edit("video run with captions", r'''cmd = build_command(exe, src, dst, o)
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=MAX_SECONDS)''', r'''srt_text = request.form.get("srt", "")[:60000].strip()
        if srt_text and fmt == "mp4" and not o["reverse"]:
            with open(os.path.join(work, "cap.srt"), "w", encoding="utf-8") as sf:
                sf.write(srt_text + "\n")
            o["srt"] = "cap.srt"
        cmd = build_command(exe, src, dst, o)
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=MAX_SECONDS, cwd=work)
            if p.returncode != 0 and o.get("srt"):  # the server may lack fonts for subtitles: try again without them
                o.pop("srt")
                skipped = True
                cmd = build_command(exe, src, dst, o)
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=MAX_SECONDS, cwd=work)''')
edit("video subtitle filter", r'''cmd += ["-vf", ",".join(vf)]''', r'''if o.get("srt"):
        vf.append("subtitles=" + o["srt"] + ":force_style='Fontsize=20,Outline=2,Shadow=0,MarginV=24'")
    cmd += ["-vf", ",".join(vf)]''')
edit("video header", r'''return send_file(io.BytesIO(data), mimetype=mime, as_attachment=False, download_name=name)''',
     r'''resp = send_file(io.BytesIO(data), mimetype=mime, as_attachment=False, download_name=name)
    if skipped:
        resp.headers["X-Captions-Skipped"] = "1"
    return resp''')

CAPTION_ROUTE = r'''def _srt_time(t):
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return "%02d:%02d:%02d,%03d" % (h, m, s, ms)


def _make_srt(segs, speed=1.0):
    out, n = [], 0
    for sg in segs:
        def g(k, sg=sg):
            return sg.get(k) if isinstance(sg, dict) else getattr(sg, k, None)
        txt = str(g("text") or "").strip()
        try:
            a, b = float(g("start")), float(g("end"))
        except (TypeError, ValueError):
            continue
        if not txt or b <= a:
            continue
        n += 1
        out.append("%d\n%s --> %s\n%s\n" % (n, _srt_time(a / speed), _srt_time(b / speed), txt))
    return "\n".join(out)


@app.post("/video/captions")
def video_captions_route():
    """Speech to text for a video (Whisper). Returns subtitles in SRT form."""
    if denied():
        return jsonify(error="locked"), 401
    exe = ffmpeg_path()
    if not exe:
        return jsonify(error="Video tools are not installed on this server."), 503
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="Choose a video file first."), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in VIDEO_EXT:
        return jsonify(error="Use an MP4, MOV, WebM, MKV, AVI or 3GP video."), 400
    start, end = _num("start", 0, 0, 36000), _num("end", 0, 0, 36000)
    speed = round(_num("speed", 1, 0.5, 2), 2)
    if not _video_slot.acquire(blocking=False):
        return jsonify(error="Another video is being processed. Try again in a minute."), 503
    work = tempfile.mkdtemp(prefix="oq_")
    try:
        src, mp3 = os.path.join(work, "in" + ext), os.path.join(work, "a.mp3")
        f.save(src)
        cmd = [exe, "-y", "-hide_banner", "-loglevel", "error", "-nostdin"]
        if start > 0:
            cmd += ["-ss", "%.2f" % start]
        if end > start:
            cmd += ["-t", "%.2f" % (end - start)]
        cmd += ["-i", src, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "32k", mp3]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=MAX_SECONDS)
        except subprocess.TimeoutExpired:
            return jsonify(error="This took too long. Try a shorter clip."), 504
        if p.returncode != 0 or not os.path.exists(mp3) or os.path.getsize(mp3) < 1000:
            return jsonify(error="Could not read any sound from this video."), 422
        if os.path.getsize(mp3) > 24 * 1024 * 1024:
            return jsonify(error="The sound is too long to caption. Trim the video to under 10 minutes."), 413
        with open(mp3, "rb") as fh:
            data = fh.read()
    finally:
        shutil.rmtree(work, ignore_errors=True)
        _video_slot.release()
    resp = None
    for model in WHISPER_MODELS:
        try:
            resp = client.audio.transcriptions.create(file=("audio.mp3", data), model=model, temperature=0,
                                                      response_format="verbose_json")
            break
        except Exception as e:
            print("captions", model, "failed:", str(e)[:160])
    if resp is None:
        return jsonify(error="Could not write captions right now. Try again in a minute."), 502
    segs = getattr(resp, "segments", None)
    if segs is None and isinstance(resp, dict):
        segs = resp.get("segments")
    srt = _make_srt(segs or [], speed)
    if not srt:
        return jsonify(error="No speech was found in this video."), 422
    return jsonify(srt=srt)


'''
edit("captions route", r'''@app.post("/video")''', CAPTION_ROUTE + r'''@app.post("/video")''')

# ----------------------------------------------------------------- 3. VIDEO CAPTIONS (page)
edit("captions html", r'''<div class="actions"><button class="primary" id="v-go" type="button">Edit video</button>''',
     r'''<div class="grp" data-fmt="mp4"><h3>Captions</h3>
            <div class="actions"><button class="btn" id="v-capgo" type="button">Make captions</button><a class="btn" id="v-capdl" download="captions.srt" hidden>Download SRT</a></div>
            <textarea class="in" id="v-srt" rows="6" placeholder="Captions appear here. You can fix any line." hidden></textarea>
            <label style="display:flex;gap:10px;align-items:center;font-family:var(--ui);font-size:15px"><input type="checkbox" id="v-capburn" checked> Put the captions on the video</label>
            <p class="muted" id="v-capstat" aria-live="polite"></p></div>
          <div class="actions"><button class="primary" id="v-go" type="button">Edit video</button>''')
edit("captions code", r'''$('#v-go').onclick=()=>{if(!vfile)return;''', r'''function srtLink(){
  const a=$('#v-capdl'),t=$('#v-srt').value;
  if(!t.trim()){a.hidden=true;return}
  if(a.dataset.u)URL.revokeObjectURL(a.dataset.u);
  const u=URL.createObjectURL(new Blob([t],{type:'text/plain'}));a.dataset.u=u;a.href=u;a.hidden=false;
}
$('#v-srt').addEventListener('input',srtLink);
$('#vsrc').addEventListener('loadstart',()=>{$('#v-srt').value='';$('#v-srt').hidden=true;$('#v-capdl').hidden=true;$('#v-capstat').textContent=''});
$('#v-capgo').onclick=async()=>{
  if(!vfile)return;
  const st=$('#v-capstat'),b=$('#v-capgo');
  st.textContent='Listening to the video and writing captions. This can take a minute...';b.disabled=true;
  const go=()=>{
    const fd=new FormData();fd.append('file',vfile);fd.append('start',$('#v-start').value||0);
    fd.append('end',$('#v-end').value||0);fd.append('speed',$('#v-speed').value);
    return fetch('/video/captions',{method:'POST',headers:{'X-Access-Key':localStorage.ak||''},body:fd});
  };
  try{
    let r=await go();
    if(r.status===401){localStorage.ak=prompt('Enter the access key')||'';r=await go()}
    const j=await r.json().catch(()=>({}));
    if(!r.ok||j.error)throw new Error(j.error||'Could not write captions.');
    $('#v-srt').hidden=false;$('#v-srt').value=j.srt;srtLink();
    st.textContent='Captions ready. Fix any line you like, then press Edit video. Make them again if you change the trim or speed.';
  }catch(e){st.textContent=e.message}
  b.disabled=false;
};
$('#v-go').onclick=()=>{if(!vfile)return;''')
edit("captions send", r'''fd.append('fade',$('#v-fade').checked?'1':'0');''',
     r'''fd.append('fade',$('#v-fade').checked?'1':'0');
  if(vfmt==='mp4'&&$('#v-capburn').checked&&$('#v-srt').value.trim())fd.append('srt',$('#v-srt').value);''')
edit("captions done note", r'''$('#vresult').hidden=false;vstatus('Done.');''',
     r'''$('#vresult').hidden=false;vstatus(x.getResponseHeader('X-Captions-Skipped')?'Done, but this server could not put the captions on the video. Use the Download SRT button instead.':'Done.');''')


def main():
    with open(FILE, "r", encoding="utf-8") as f:
        src = f.read()
    if "IMG_STYLES" in src and "video_captions_route" in src:
        sys.exit("Already patched. Nothing to do.")
    for name, old, new in EDITS:
        n = src.count(old)
        if n != 1:
            sys.exit("Change '%s' matched %d times (needs exactly 1) in %s. Nothing was changed." % (name, n, FILE))
        src = src.replace(old, new)
    try:
        compile(src, FILE, "exec")
    except SyntaxError as e:
        sys.exit("The patched file has a syntax error (line %s: %s). Nothing was changed." % (e.lineno, e.msg))
    shutil.copyfile(FILE, FILE + ".bak_media")
    with open(FILE, "w", encoding="utf-8") as f:
        f.write(src)
    print("Patched %s (backup: %s.bak_media). Commit and push to deploy." % (FILE, FILE))


if __name__ == "__main__":
    main()
