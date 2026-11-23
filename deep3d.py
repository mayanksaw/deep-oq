"""
deep3d.py - everything for the Deep OQ 3D tab in ONE file.

Install:  pip install trimesh numpy requests gradio_client
Optional env var on Render:  MESHY_API_KEY   (Meshy needs a PAID plan for API use)

QUICK mode  : the AI writes a list of shapes (or a hand-made template is used) -> rounded 3D model.
AI MESH mode: real generated mesh. Free photo->3D via TripoSR on Hugging Face, or text->3D via Meshy
              when MESHY_API_KEY is set. If it fails, it falls back to Quick mode.

Main entry points (use these in your Flask routes):
  start_job(prompt, mode="quick"|"ai", image_path=None, ask_llm=None, text_to_image=None) -> job_id
  job_status(job_id) -> {"status": "working"|"done"|"error", "path":..., "note":..., "error":...}
  convert(path, "glb"|"stl"|"obj") -> bytes for the download buttons

Quick-mode coordinates: Y is up, the model faces +Z, X is left/right, feet sit at y = 0.
Part types: box, sphere, ellipsoid, cylinder, capsule, cone, limb(a,b,r). Optional: rot, color, mirror.
"""
import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid

import numpy as np
import requests
import trimesh
from trimesh import transformations as tf

OUT_DIR = os.path.join(tempfile.gettempdir(), "deepoq_meshes")
os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================================
#  PART 1 - QUICK MODE (shape builder)
# =====================================================================
MAX_PARTS = 80

NAMED_COLORS = {
    "red": "#c0392b", "green": "#27ae60", "blue": "#2980b9", "yellow": "#f1c40f",
    "orange": "#e67e22", "purple": "#8e44ad", "pink": "#e91e8c", "brown": "#8b5a2b",
    "black": "#1a1a1a", "white": "#f2f2f2", "gray": "#7f8c8d", "grey": "#7f8c8d",
    "gold": "#d4af37", "silver": "#c0c0c0", "tan": "#d2b48c", "cream": "#f5e6c8",
}


# ----------------------------------------------------------------- helpers
def _num(v, default=0.0):
    try:
        v = float(v)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _vec(v, n=3, default=0.0):
    if not isinstance(v, (list, tuple)):
        v = []
    v = [_num(x, default) for x in v][:n]
    return v + [default] * (n - len(v))


def _rgba(color):
    c = str(color or "#9aa0a6").strip().lower()
    c = NAMED_COLORS.get(c, c)
    if re.fullmatch(r"#[0-9a-f]{6}", c):
        return [int(c[i:i + 2], 16) for i in (1, 3, 5)] + [255]
    return [154, 160, 166, 255]


def _align(src, dst):
    """4x4 matrix rotating direction src onto direction dst."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    return trimesh.geometry.align_vectors(src / np.linalg.norm(src), dst / np.linalg.norm(dst))


_Z_TO_Y = _align([0, 0, 1], [0, 1, 0])


def _euler(rot):
    rx, ry, rz = np.radians(_vec(rot))
    return tf.euler_matrix(rx, ry, rz, "sxyz")


# ----------------------------------------------------------------- one part -> mesh
def _make_part(p, mirrored=False):
    t = str(p.get("type", "box")).lower()
    pos = _vec(p.get("pos"))
    size = [max(abs(x), 0.01) for x in _vec(p.get("size"), 3, 0.3)]
    r = max(abs(_num(p.get("r"), 0.1)), 0.005)
    h = max(abs(_num(p.get("h"), 0.3)), 0.01)

    if t == "box":
        m = trimesh.creation.box(extents=size)
    elif t == "sphere":
        m = trimesh.creation.icosphere(subdivisions=3, radius=r)
    elif t == "ellipsoid":
        m = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
        m.apply_scale(size)
    elif t == "cylinder":
        m = trimesh.creation.cylinder(radius=r, height=h, sections=24)
        m.apply_transform(_Z_TO_Y)
    elif t == "capsule":
        m = trimesh.creation.capsule(height=max(h - 2 * r, 0.001), radius=r, count=[12, 24])
        m.apply_transform(_Z_TO_Y)
    elif t == "cone":
        m = trimesh.creation.cone(radius=r, height=h, sections=24)
        m.apply_translation([0, 0, -h / 2])  # centre it
        m.apply_transform(_Z_TO_Y)
    elif t == "limb":
        a = np.array(_vec(p.get("a")))
        b = np.array(_vec(p.get("b"), 3, 0.0))
        d = b - a
        length = float(np.linalg.norm(d))
        if length < 1e-6:
            return None
        m = trimesh.creation.capsule(height=max(length - 2 * r, 0.001), radius=r, count=[12, 24])
        m.apply_transform(_align([0, 0, 1], d))
        m.apply_translation((a + b) / 2)
        pos = [0, 0, 0]  # limb is already placed using a and b
    else:
        return None

    m.apply_transform(_euler(p.get("rot")))
    m.apply_translation(pos)

    if mirrored:  # reflect across x = 0 (flip winding so faces still point outwards)
        m.apply_transform(np.diag([-1.0, 1.0, 1.0, 1.0]))
        m.invert()

    m.visual.face_colors = np.tile(_rgba(p.get("color")), (len(m.faces), 1))
    return m


# ----------------------------------------------------------------- whole model
def build_mesh(parts):
    """parts: list of dicts. Returns one trimesh.Trimesh sitting on y = 0."""
    meshes = []
    for p in list(parts)[:MAX_PARTS]:
        if not isinstance(p, dict):
            continue
        for mirrored in ([False, True] if p.get("mirror") else [False]):
            try:
                m = _make_part(p, mirrored)
            except Exception as e:  # one bad part must not kill the model
                print("quick3d: skipped part", p.get("type"), str(e)[:120])
                continue
            if m is not None:
                meshes.append(m)
    if not meshes:
        raise ValueError("no valid parts")
    model = trimesh.util.concatenate(meshes)
    model.apply_translation([0, -model.bounds[0][1], 0])  # drop onto the ground
    return model


def export_bytes(model, fmt="glb"):
    """fmt: glb, stl or obj -> bytes ready to send as a download."""
    fmt = fmt.lower()
    if fmt not in ("glb", "stl", "obj"):
        raise ValueError("format must be glb, stl or obj")
    data = model.export(file_type=fmt)
    return data.encode("utf-8") if isinstance(data, str) else data


# ----------------------------------------------------------------- hand-made templates
def dog_parts(color="#a0662f", dark="#2b1d12"):
    return [
        {"type": "ellipsoid", "pos": [0, 0.68, 0.0], "size": [0.46, 0.42, 0.95], "color": color},   # body
        {"type": "ellipsoid", "pos": [0, 0.74, 0.30], "size": [0.44, 0.44, 0.45], "color": color},  # chest
        {"type": "limb", "a": [0, 0.78, 0.42], "b": [0, 0.92, 0.58], "r": 0.12, "color": color},    # neck
        {"type": "ellipsoid", "pos": [0, 0.95, 0.66], "size": [0.34, 0.31, 0.38], "color": color},  # head
        {"type": "ellipsoid", "pos": [0, 0.89, 0.88], "size": [0.19, 0.16, 0.26], "color": "#c58a50"},  # snout
        {"type": "sphere", "pos": [0, 0.91, 1.02], "r": 0.04, "color": dark},                       # nose
        {"type": "sphere", "pos": [0.09, 1.0, 0.82], "r": 0.025, "color": dark, "mirror": True},    # eyes
        {"type": "ellipsoid", "pos": [0.14, 1.04, 0.58], "size": [0.1, 0.22, 0.16],
         "rot": [0, 0, -20], "color": dark, "mirror": True},                                         # ears
        {"type": "limb", "a": [0.15, 0.55, 0.30], "b": [0.15, 0.05, 0.32], "r": 0.065,
         "color": color, "mirror": True},                                                            # front legs
        {"type": "limb", "a": [0.15, 0.58, -0.30], "b": [0.15, 0.05, -0.30], "r": 0.075,
         "color": color, "mirror": True},                                                            # back legs
        {"type": "ellipsoid", "pos": [0.15, 0.04, 0.35], "size": [0.14, 0.08, 0.2],
         "color": "#c58a50", "mirror": True},                                                        # front paws
        {"type": "ellipsoid", "pos": [0.15, 0.04, -0.27], "size": [0.14, 0.08, 0.2],
         "color": "#c58a50", "mirror": True},                                                        # back paws
        {"type": "limb", "a": [0, 0.82, -0.45], "b": [0, 1.15, -0.72], "r": 0.04, "color": color},  # tail
    ]


def chair_parts(color="#8b5a2b"):
    legs = [{"type": "box", "pos": [0.22, 0.22, 0.22], "size": [0.06, 0.44, 0.06], "color": color, "mirror": True},
            {"type": "box", "pos": [0.22, 0.22, -0.22], "size": [0.06, 0.44, 0.06], "color": color, "mirror": True}]
    return legs + [
        {"type": "box", "pos": [0, 0.47, 0], "size": [0.55, 0.06, 0.55], "color": color},          # seat
        {"type": "box", "pos": [0, 0.82, -0.25], "size": [0.55, 0.64, 0.05], "color": color},      # back rest
    ]


def snowman_parts():
    return [
        {"type": "sphere", "pos": [0, 0.35, 0], "r": 0.35, "color": "white"},
        {"type": "sphere", "pos": [0, 0.88, 0], "r": 0.26, "color": "white"},
        {"type": "sphere", "pos": [0, 1.3, 0], "r": 0.19, "color": "white"},
        {"type": "cone", "pos": [0, 1.3, 0.24], "r": 0.04, "h": 0.2, "rot": [90, 0, 0], "color": "orange"},
        {"type": "sphere", "pos": [0.07, 1.36, 0.16], "r": 0.02, "color": "black", "mirror": True},
        {"type": "cylinder", "pos": [0, 1.52, 0], "r": 0.12, "h": 0.2, "color": "black"},
        {"type": "cylinder", "pos": [0, 1.43, 0], "r": 0.22, "h": 0.03, "color": "black"},
    ]


TEMPLATES = {
    "dog": dog_parts, "puppy": dog_parts,
    "chair": chair_parts,
    "snowman": snowman_parts,
}


def find_template(text):
    words = re.findall(r"[a-z]+", text.lower())
    for w in words:
        if w in TEMPLATES:
            return TEMPLATES[w]
    return None


# ----------------------------------------------------------------- prompt for your AI
SHAPE_SYSTEM_PROMPT = """You are a 3D modeller. Reply with ONLY a JSON object {"parts":[...]} - no words, no code fences.

Coordinates: Y is up, the model faces +Z, X is left/right, feet rest at y=0. Total height about 1 unit (people 1.7, buildings 3-5).
Part types:
 box {pos,size:[w,h,d]}   sphere {pos,r}   ellipsoid {pos,size:[w,h,d]}
 cylinder {pos,r,h}   capsule {pos,r,h}   cone {pos,r,h}  (cylinder, capsule and cone stand upright along Y)
 limb {a:[x,y,z], b:[x,y,z], r}  = rounded tube from point a to point b. Use for legs, arms, tails, branches, necks.
Every part may also have: rot:[rx,ry,rz] degrees, color:"#rrggbb", mirror:true (adds a copy at -x, so describe only one side).

Rules:
1. Use 15 to 40 parts. Prefer ellipsoid, sphere and limb for living things; they look organic. Use box only for man-made objects.
2. Parts must touch or overlap their neighbours. No gaps: legs overlap the body, the head overlaps the neck.
3. Add small details (eyes, nose, ears, paws, handles, windows) with a darker or different colour.
4. Use 3 or 4 harmonious colours.
5. Use mirror:true for anything that comes in pairs.

Example for "mushroom":
{"parts":[
 {"type":"cylinder","pos":[0,0.2,0],"r":0.12,"h":0.4,"color":"#f1e6d0"},
 {"type":"ellipsoid","pos":[0,0.45,0],"size":[0.8,0.45,0.8],"color":"#c0392b"},
 {"type":"sphere","pos":[0.18,0.62,0.1],"r":0.05,"color":"#ffffff","mirror":true}]}
"""


def parse_parts(llm_text):
    """Pull the JSON out of an AI reply, even if it added extra words or ``` fences."""
    s = re.sub(r"```(?:json)?", "", str(llm_text))
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("no JSON in reply")
    data = json.loads(s[a:b + 1])
    parts = data.get("parts") if isinstance(data, dict) else data
    if not isinstance(parts, list) or not parts:
        raise ValueError("no parts in reply")
    return parts


def make_quick_model(user_text, ask_llm=None):
    """
    user_text : what the person typed, e.g. "dog"
    ask_llm   : function(system_prompt, user_text) -> str   (plug in your own Groq/Claude call)
    Returns a trimesh model. Uses a hand-made template if one matches, otherwise the AI.
    """
    tpl = find_template(user_text)
    if tpl:
        return build_mesh(tpl())
    if ask_llm is None:
        raise ValueError("no template for this and no AI function given")
    last = None
    for _ in range(2):  # one retry if the AI sends bad JSON
        try:
            return build_mesh(parse_parts(ask_llm(SHAPE_SYSTEM_PROMPT, user_text)))
        except Exception as e:
            last = e
    raise RuntimeError(f"could not build model: {last}")



# =====================================================================
#  PART 2 - AI MESH MODE (real generated models)
# =====================================================================
MESHY_KEY = os.environ.get("MESHY_API_KEY", "").strip()
MESHY_URL = "https://api.meshy.ai/openapi/v2/text-to-3d"


# ------------------------------------------------------------------ Meshy (text -> 3D)
def _meshy_wait(task_id, headers, timeout=600):
    end = time.time() + timeout
    while time.time() < end:
        r = requests.get(f"{MESHY_URL}/{task_id}", headers=headers, timeout=30)
        r.raise_for_status()
        task = r.json()
        if task.get("status") == "SUCCEEDED":
            return task
        if task.get("status") in ("FAILED", "CANCELED"):
            raise RuntimeError("Meshy task failed: " + str(task.get("task_error"))[:200])
        time.sleep(5)
    raise TimeoutError("Meshy took too long")


def meshy_text_to_3d(prompt, textured=True):
    headers = {"Authorization": f"Bearer {MESHY_KEY}"}
    r = requests.post(MESHY_URL, headers=headers, timeout=30,
                      json={"mode": "preview", "prompt": prompt, "should_remesh": True})
    r.raise_for_status()
    task = _meshy_wait(r.json()["result"], headers)          # untextured shape
    if textured:
        r = requests.post(MESHY_URL, headers=headers, timeout=30,
                          json={"mode": "refine", "preview_task_id": task["id"], "enable_pbr": True})
        r.raise_for_status()
        task = _meshy_wait(r.json()["result"], headers)      # adds colours / materials
    url = (task.get("model_urls") or {}).get("glb")
    if not url:
        raise RuntimeError("Meshy returned no GLB link")
    path = os.path.join(OUT_DIR, uuid.uuid4().hex + ".glb")
    with open(path, "wb") as f:
        f.write(requests.get(url, timeout=120).content)
    return path


# ------------------------------------------------------------------ TripoSR (photo -> 3D, free)
def tripo_from_image(image_path, mc_resolution=256):
    from gradio_client import Client
    try:
        from gradio_client import handle_file
    except ImportError:                      # older gradio_client versions
        from gradio_client import file as handle_file

    client = Client("stabilityai/TripoSR")
    processed = client.predict(handle_file(image_path), True, 0.85, api_name="/preprocess")
    if not isinstance(processed, str):
        processed = processed[0] if isinstance(processed, (list, tuple)) else processed
    result = client.predict(handle_file(processed), mc_resolution, api_name="/generate")

    files = result if isinstance(result, (list, tuple)) else [result]
    files = [f for f in files if isinstance(f, str)]
    if not files:
        raise RuntimeError("TripoSR returned no file")
    pick = (next((f for f in files if f.lower().endswith(".glb")), None)
            or next((f for f in files if f.lower().endswith(".obj")), files[0]))
    path = os.path.join(OUT_DIR, uuid.uuid4().hex + os.path.splitext(pick)[1])
    shutil.copy(pick, path)
    return path


# ------------------------------------------------------------------ chooser
def make_ai_mesh(prompt="", image_path=None, text_to_image=None):
    """
    prompt         : text description (optional if image_path is given)
    image_path     : path of an uploaded photo (optional)
    text_to_image  : optional function(prompt) -> image file path. Plug in your own Photo
                     generator so text-only requests can still use the free TripoSR engine.
    Returns the path of a .glb (or .obj) file. Raises RuntimeError with a readable message.
    """
    errors = []
    if not image_path and MESHY_KEY and prompt:
        try:
            return meshy_text_to_3d(prompt)
        except Exception as e:
            errors.append("Meshy: " + str(e)[:150])
    if not image_path and prompt and text_to_image:
        try:
            image_path = text_to_image(prompt + ", single object, centered, plain white background, studio lighting")
        except Exception as e:
            errors.append("Image step: " + str(e)[:150])
    if image_path:
        try:
            return tripo_from_image(image_path)
        except Exception as e:
            errors.append("TripoSR: " + str(e)[:150])
    if not errors:
        errors.append("Give a photo, or set MESHY_API_KEY, or pass a text_to_image function.")
    raise RuntimeError(" | ".join(errors))


def convert(path, fmt="glb"):
    """Convert whatever file we got into glb / stl / obj bytes for the download buttons."""
    import trimesh
    mesh = trimesh.load(path, force="mesh")
    data = mesh.export(file_type=fmt.lower())
    return data.encode("utf-8") if isinstance(data, str) else data



# =====================================================================
#  PART 3 - ONE ENTRY POINT + BACKGROUND JOBS
# =====================================================================
def make_model_file(prompt="", mode="quick", image_path=None, ask_llm=None,
                    text_to_image=None, fallback=True):
    """Returns (path_to_glb_or_obj, note). note is "" unless a fallback happened."""
    note = ""
    if mode == "ai":
        try:
            return make_ai_mesh(prompt, image_path, text_to_image), note
        except Exception as e:
            if not (fallback and prompt):
                raise
            print("AI mesh failed, using Quick mode:", str(e)[:200])
            note = "AI mesh was not available, so this is a Quick-mode model."
    model = make_quick_model(prompt, ask_llm)
    path = os.path.join(OUT_DIR, uuid.uuid4().hex + ".glb")
    with open(path, "wb") as f:
        f.write(export_bytes(model, "glb"))
    return path, note


JOBS = {}   # kept in memory: run gunicorn with ONE worker (e.g. --workers 1 --threads 4)


def start_job(prompt="", mode="quick", image_path=None, ask_llm=None, text_to_image=None):
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "working"}

    def run():
        try:
            path, note = make_model_file(prompt, mode, image_path, ask_llm, text_to_image)
            JOBS[job_id] = {"status": "done", "path": path, "note": note}
        except Exception as e:
            JOBS[job_id] = {"status": "error", "error": str(e)[:400]}

    threading.Thread(target=run, daemon=True).start()
    return job_id


def job_status(job_id):
    return JOBS.get(job_id, {"status": "error", "error": "unknown job"})


if __name__ == "__main__":
    path, _ = make_model_file("dog", mode="quick")
    for ext in ("glb", "stl", "obj"):
        with open(f"dog.{ext}", "wb") as f:
            f.write(convert(path, ext))
    print("wrote dog.glb, dog.stl, dog.obj")
