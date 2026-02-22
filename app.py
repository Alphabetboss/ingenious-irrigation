from __future__ import annotations

# app.py — Ingenious Irrigation: UI + schedule + commands + Astra chat + Overwatch loop (offline-first)
# Works standalone with fallbacks; auto-hooks to optional modules if present:
#   - schedule_manager.py (start/stop/status/set duration + plan/mark_ran)
#   - hydration_engine.py (HydrationEngine, Inputs)
#   - health_evaluator.py (HealthEvaluator)

from pathlib import Path
import os
import json
import time
import socket
import threading
import re
from typing import Any, Dict, Optional

from flask import Flask, render_template, send_from_directory, request, jsonify


# ---------------------------
# Paths & Flask
# ---------------------------
ROOT = Path(__file__).parent
TPL = ROOT / "templates"
STATIC = ROOT / "static"
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=str(STATIC), template_folder=str(TPL))
app.config["TEMPLATES_AUTO_RELOAD"] = True

# ---------------------------
# Files / Globals
# ---------------------------
SCHEDULE_JSON = DATA / "schedule.json"
DEFAULT_SCHEDULE = {"zones": {"1": {"minutes": 10, "enabled": True}}}

WEATHER_JSON = DATA / "weather_cache.json"
ALERT_QUEUE = DATA / "alert_queue.json"
HYDRATION_CSV = DATA / "hydration_scores.csv"
SENSOR_CSV = DATA / "sensor_log.csv"
WATERING_LOG = DATA / "watering.log"

CURRENT_WATERING: Dict[str, Any] = {"active": False, "zone": None, "since": None}
ONLINE_STATE: Dict[str, Any] = {"online": False, "last_check": 0}
ps_start = time.time()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()


# ---------------------------
# Small safe helpers
# ---------------------------
def _json_read(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _json_write(path: Path, payload: Any) -> None:
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def _clamp_int(val, default=10, lo=0, hi=240) -> int:
    try:
        n = int(val)
    except (TypeError, ValueError):
        n = default
    return max(lo, min(hi, n))


def log_event(path: Path, line: str) -> None:
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")
    except Exception:
        pass


# ---------------------------
# Online check (cached)
# ---------------------------
def is_online(ttl: int = 10) -> bool:
    now = time.time()
    if now - ONLINE_STATE.get("last_check", 0) < ttl:
        return bool(ONLINE_STATE.get("online", False))

    ONLINE_STATE["last_check"] = now
    try:
        socket.setdefaulttimeout(1.5)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # DNS port sometimes blocked on some networks; still OK as a fast check
            s.connect(("1.1.1.1", 53))
        ONLINE_STATE["online"] = True
    except Exception:
        ONLINE_STATE["online"] = False

    return bool(ONLINE_STATE["online"])


# ---------------------------
# Schedule helpers
# ---------------------------
def load_schedule() -> dict:
    if not SCHEDULE_JSON.exists():
        _json_write(SCHEDULE_JSON, DEFAULT_SCHEDULE)
        return DEFAULT_SCHEDULE.copy()

    data = _json_read(SCHEDULE_JSON, DEFAULT_SCHEDULE.copy())
    # normalize
    if not isinstance(data, dict):
        data = DEFAULT_SCHEDULE.copy()
    data.setdefault("zones", {})
    if not isinstance(data["zones"], dict):
        data["zones"] = {}
    if "1" not in data["zones"]:
        data["zones"]["1"] = {"minutes": 10, "enabled": True}
    return data


def save_schedule(d: dict) -> None:
    _json_write(SCHEDULE_JSON, d)


# ---------------------------
# Weather helpers (safe offline)
# ---------------------------
def get_weather_safe() -> dict:
    """
    Returns a minimal weather dict and caches it.
    Replace the TODO with a real API later.
    """
    fallback = {"temp_f": None, "humidity": None, "rain_in": None, "source": "cache/off"}

    if is_online():
        try:
            # TODO: fetch real data here
            w = dict(fallback)
            w["source"] = "online"
            _json_write(WEATHER_JSON, w)
            return w
        except Exception:
            pass

    cached = _json_read(WEATHER_JSON, None)
    if isinstance(cached, dict):
        cached.setdefault("source", "cache/off")
        return cached

    return fallback


def weather_overrides() -> dict:
    """
    Decide whether to skip/boost watering using simple heuristics.
    """
    w = get_weather_safe()
    temp = (w.get("temp_f") if w.get("temp_f") is not None else 75)
    rain = (w.get("rain_in") if w.get("rain_in") is not None else 0.0)

    skip = False
    reason = None
    boost = 0

    if rain >= 0.10:
        skip, reason = True, "Rain ≥ 0.1in"
    elif temp < 40:
        skip, reason = True, "Below 40°F"
    elif temp > 93:
        boost, reason = 5, "Hot day > 93°F (boost 5 min)"

    return {
        "ok": True,
        "source": w.get("source", "cache/off"),
        "temp_f": temp,
        "rain_in": rain,
        "skip": skip,
        "boost_min": boost,
        "reason": reason,
    }


# ---------------------------
# Astra chat (offline-first)
# ---------------------------
def offline_reply(user_text: str) -> str:
    t = (user_text or "").lower().strip()
    if not t:
        return "Astra here. Ask about status, watering, schedule, leaks, or weather."
    if "status" in t:
        s = get_status()
        if s.get("active") or s.get("watering"):
            return f"System OK. Zone {s.get('zone')} is running."
        return "System OK. All zones idle."
    if "weather" in t or "rain" in t or "forecast" in t:
        d = weather_overrides()
        if d["skip"]:
            return f"Weather override: skipping. Reason: {d['reason']}."
        if d["boost_min"]:
            return f"Weather override: boosting +{d['boost_min']} minutes. Reason: {d['reason']}."
        return "Weather looks normal. No overrides."
    if "start" in t or "water" in t or "run" in t:
        return "Say: “start zone 1 for 10 minutes” or press Start in the dashboard."
    if "stop" in t:
        return "Say “stop watering” or press Stop."
    if "schedule" in t or "timer" in t:
        sch = load_schedule()
        z1 = sch["zones"].get("1", {})
        return f"Zone 1 is set to {int(z1.get('minutes', 10))} minutes. Want to change it?"
    if "leak" in t or "burst" in t:
        return "Overwatch monitors for pooling water/mud. If detected, I’ll alert and log it."
    return "I can start/stop watering, adjust minutes, check status, and apply weather overrides."


def online_reply(messages: list[dict], timeout: int = 8) -> str:
    """
    Uses OpenAI Chat Completions (legacy-style endpoint).
    If it fails, falls back to offline.
    """
    if not OPENAI_API_KEY:
        return offline_reply(messages[-1]["content"])

    try:
        import requests

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": 220,
        }

        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        text = (data["choices"][0]["message"]["content"] or "").strip()
        return text or offline_reply(messages[-1]["content"])
    except Exception:
        return offline_reply(messages[-1]["content"])


# ---------------------------
# Minimal camera / detection stubs (replace later)
# ---------------------------
def capture_image() -> Path:
    # TODO: hook to Pi camera; keep path stable
    return DATA / "last.jpg"


def yolo_detect(img_path: Path) -> dict:
    # TODO: call your real model; keep output keys stable
    return {"water": 0.05, "grass": 0.70, "dead_grass": 0.10, "mud": 0.15}


def queue_alert(kind: str, detail: str) -> None:
    payload = {"ts": int(time.time()), "kind": kind, "detail": detail}
    q = _json_read(ALERT_QUEUE, [])
    if not isinstance(q, list):
        q = []
    q.append(payload)
    _json_write(ALERT_QUEUE, q)


def try_send_alerts_online() -> None:
    # placeholder sender: clears queue when online
    if not is_online() or not ALERT_QUEUE.exists():
        return
    q = _json_read(ALERT_QUEUE, [])
    if not isinstance(q, list) or not q:
        return
    _json_write(ALERT_QUEUE, [])


# ---------------------------
# Overwatch loop
# ---------------------------
_stop_overwatch = threading.Event()

def overwatch_loop(
    interval_sec: int = 20,
    leak_thresh: float = 0.35,
    burst_thresh: float = 0.60,
    require_consecutive: int = 2,
):
    consec_high = 0

    # ensure csv header exists
    if not HYDRATION_CSV.exists():
        log_event(HYDRATION_CSV, "ts,hydration_score")

    while not _stop_overwatch.is_set():
        try:
            img = capture_image()
            det = yolo_detect(img)

            # wetness = water + mud (0..2-ish), clamp to 0..1 for scoring
            wet_raw = float(det.get("water", 0.0)) + float(det.get("mud", 0.0))
            wet = max(0.0, min(1.0, wet_raw))

            # greenness proxy (can push score down a bit if lawn looks healthy)
            green = float(det.get("grass", 0.0)) - float(det.get("dead_grass", 0.0))
            green = max(-1.0, min(1.0, green))

            # Your scale: 0=dry(need watering) ... 10=oversaturated(no watering)
            score = 5 + (wet * 5) - (green * 2)
            score = max(0.0, min(10.0, float(score)))

            # leak detection when idle
            if not CURRENT_WATERING.get("active") and wet_raw >= leak_thresh:
                msg = f"Possible leak while idle: wet={wet_raw:.2f}"
                log_event(WATERING_LOG, f"{int(time.time())} LEAK {msg}")
                queue_alert("leak", msg)

            # burst detection
            if wet_raw >= burst_thresh:
                consec_high += 1
                if consec_high >= require_consecutive:
                    msg = f"PIPE BURST suspected: wet={wet_raw:.2f} sustained"
                    log_event(WATERING_LOG, f"{int(time.time())} PIPE {msg}")
                    queue_alert("pipe_burst", msg)
            else:
                consec_high = 0

            log_event(HYDRATION_CSV, f"{int(time.time())},{round(score, 1)}")
            try_send_alerts_online()

        except Exception as e:
            log_event(WATERING_LOG, f"{int(time.time())} OW_ERR {repr(e)}")

        _stop_overwatch.wait(interval_sec)


# Start Overwatch exactly once
_overwatch_started = False
_overwatch_lock = threading.Lock()

def start_overwatch_thread_once():
    global _overwatch_started
    with _overwatch_lock:
        if _overwatch_started:
            return
        t = threading.Thread(target=overwatch_loop, args=(20,), daemon=True)
        t.start()
        _overwatch_started = True


# ---------------------------
# Optional module hooks (safe fallbacks)
# ---------------------------
HealthEvaluator = None
HydrationEngineImported = None
InputsImported = None
build_plan_for_today = None
mark_ran_today = None
_sm_start = _sm_stop = _sm_status = _sm_set_dur = None

try:
    import importlib
    _mod = importlib.import_module("health_evaluator")
    HealthEvaluator = getattr(_mod, "HealthEvaluator", None)
except Exception:
    HealthEvaluator = None

try:
    import importlib
    _mod_h = importlib.import_module("hydration_engine")
    HydrationEngineImported = getattr(_mod_h, "HydrationEngine", None)
    InputsImported = getattr(_mod_h, "Inputs", None)
except Exception:
    HydrationEngineImported = None
    InputsImported = None

try:
    import schedule_manager as _schedule_manager
    build_plan_for_today = getattr(_schedule_manager, "build_plan_for_today", None)
    mark_ran_today = getattr(_schedule_manager, "mark_ran_today", None)
    _sm_start = getattr(_schedule_manager, "start_watering", None)
    _sm_stop = getattr(_schedule_manager, "stop_watering", None)
    _sm_status = getattr(_schedule_manager, "get_status", None)
    _sm_set_dur = getattr(_schedule_manager, "set_zone_duration", None)
except Exception:
    build_plan_for_today = None
    mark_ran_today = None
    _sm_start = _sm_stop = _sm_status = _sm_set_dur = None


# Fallback Inputs / HydrationEngine if module missing
if InputsImported is None:
    class Inputs:
        def __init__(
            self,
            soil_moisture_pct=None,
            ambient_temp_f=None,
            humidity_pct=None,
            rain_24h_in=0.0,
            rain_72h_in=0.0,
            forecast_rain_24h_in=0.0,
            greenness_score=None,
            dry_flag=False,
            water_flag=False,
        ):
            self.soil_moisture_pct = soil_moisture_pct
            self.ambient_temp_f = ambient_temp_f
            self.humidity_pct = humidity_pct
            self.rain_24h_in = rain_24h_in
            self.rain_72h_in = rain_72h_in
            self.forecast_rain_24h_in = forecast_rain_24h_in
            self.greenness_score = greenness_score
            self.dry_flag = dry_flag
            self.water_flag = water_flag
else:
    Inputs = InputsImported


if HydrationEngineImported is None:
    class HydrationEngine:
        def __init__(self, cache_file: Optional[str] = None):
            self.cache_file = cache_file

        def compute(self, inputs: Inputs):
            # fallback heuristic: midpoint + tweak from moisture flags
            score = 5.0
            try:
                if getattr(inputs, "dry_flag", False):
                    score = max(0.0, score - 2.0)
                if getattr(inputs, "water_flag", False):
                    score = min(10.0, score + 2.0)
                sm = getattr(inputs, "soil_moisture_pct", None)
                if sm is not None:
                    # higher soil moisture -> higher score (wetter)
                    score = score + ((float(sm) - 50.0) / 20.0)
            except Exception:
                pass
            score = max(0.0, min(10.0, float(score)))

            from types import SimpleNamespace
            return SimpleNamespace(
                need_score=score,
                advisory="fallback advisory (hydration_engine.py not found)",
                factors={
                    "dry_flag": bool(getattr(inputs, "dry_flag", False)),
                    "water_flag": bool(getattr(inputs, "water_flag", False)),
                    "soil_moisture_pct": getattr(inputs, "soil_moisture_pct", None),
                },
            )
else:
    HydrationEngine = HydrationEngineImported


# Schedule control fallbacks
def _stub_start_watering(zone=1, minutes=None):
    CURRENT_WATERING.update({"active": True, "zone": int(zone), "since": int(time.time())})
    if minutes is None:
        minutes = int(load_schedule()["zones"].get(str(zone), {}).get("minutes", 10))
    return {"ok": True, "watering": True, "active": True, "zone": int(zone), "minutes": int(minutes)}


def _stub_stop_watering():
    CURRENT_WATERING.update({"active": False, "zone": None, "since": None})
    return {"ok": True, "watering": False, "active": False}


def _stub_get_status():
    d = dict(CURRENT_WATERING)
    d["ok"] = True
    d.setdefault("watering", bool(d.get("active", False)))
    return d


def _stub_set_zone_duration(zone, minutes):
    data = load_schedule()
    z = str(zone)
    data.setdefault("zones", {})
    data["zones"].setdefault(z, {"enabled": True, "minutes": 10})
    data["zones"][z]["minutes"] = int(minutes)
    save_schedule(data)
    return {"ok": True, "zone": int(zone), "minutes": int(minutes)}


start_watering = _sm_start or _stub_start_watering
stop_watering = _sm_stop or _stub_stop_watering
get_status = _sm_status or _stub_get_status
set_zone_duration = _sm_set_dur or _stub_set_zone_duration


# Optional evaluator instances
_HEALTH_MODEL_PATH = (ROOT / "models" / "hydration_model.pt")
_evaluator = None
if HealthEvaluator is not None:
    try:
        _evaluator = HealthEvaluator(model_path=str(_HEALTH_MODEL_PATH) if _HEALTH_MODEL_PATH.exists() else None)
    except Exception:
        _evaluator = None

_hydrator = None
try:
    _hydrator = HydrationEngine(cache_file=str(DATA / "hydration_cache.json"))
except Exception:
    _hydrator = None


def _err(msg: str, code: int = 400):
    return jsonify({"ok": False, "error": msg}), code


# ---------------------------
# Routes
# ---------------------------
@app.after_request
def _no_cache(resp):
    ct = resp.headers.get("Content-Type", "")
    if "text/html" in ct:
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
    return resp


@app.get("/")
def dashboard():
    return render_template("dashboard.html")


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.get("/status")
def status():
    st = get_status()
    return jsonify({
        "ok": True,
        "net": "ONLINE" if is_online() else "OFFLINE",
        "uptime_sec": int(time.time() - ps_start),
        "watering": bool(st.get("active") or st.get("watering")),
        "zone": st.get("zone"),
        "since": st.get("since"),
    })


@app.get("/favicon.ico")
def favicon():
    fav_path = STATIC / "favicon.ico"
    if fav_path.exists():
        return send_from_directory(str(STATIC), "favicon.ico")
    return ("", 204)


@app.get("/api/weather_decision")
def api_weather_decision():
    return jsonify(weather_overrides())


@app.post("/api/chat")
def api_chat():
    body = request.get_json(silent=True) or {}
    user_msg = (body.get("message") or "").strip()

    if is_online():
        messages = [
            {"role": "system", "content": "You are Astra, a helpful sprinkler technician assistant. Keep replies short and practical."},
            {"role": "user", "content": user_msg or "Say hello"},
        ]
        text = online_reply(messages)
    else:
        text = offline_reply(user_msg)

    return jsonify({"ok": True, "reply": text})


# Back-compat
@app.post("/chat")
def chat():
    return api_chat()


@app.post("/api/command")
def api_command():
    """
    JSON: {"command": "start zone 1 for 5 minutes"}
    """
    body = request.get_json(silent=True) or {}
    cmd_raw = (body.get("command") or "").strip()
    cmd = cmd_raw.lower()

    if not cmd:
        return jsonify({"ok": False, "reply": "I didn’t catch that."})

    # zone detection
    zmatch = re.search(r"zone\s*(\d+)", cmd)
    zone = int(zmatch.group(1)) if zmatch else 1

    # minutes detection
    mmatch = re.search(r"(\d+)\s*(min|mins|minutes)\b", cmd)
    minutes = int(mmatch.group(1)) if mmatch else None

    if any(k in cmd for k in ["start", "turn on", "run", "water"]):
        if minutes is not None:
            set_zone_duration(zone, minutes)
        res = start_watering(zone, minutes)
        m = res.get("minutes", minutes or 10)
        return jsonify({"ok": True, "reply": f"✅ Starting zone {zone} for {m} min.", "status": res})

    if any(k in cmd for k in ["stop", "turn off", "halt", "end"]):
        res = stop_watering()
        return jsonify({"ok": True, "reply": "🛑 Watering stopped.", "status": res})

    if "status" in cmd or "check" in cmd:
        res = get_status()
        if res.get("active") or res.get("watering"):
            return jsonify({"ok": True, "reply": f"💧 Zone {res.get('zone')} running.", "status": res})
        return jsonify({"ok": True, "reply": "🌿 All zones idle.", "status": res})

    if "set" in cmd and ("minute" in cmd or "minutes" in cmd):
        if minutes is None:
            return jsonify({"ok": False, "reply": "Tell me how many minutes to set."})
        set_zone_duration(zone, minutes)
        return jsonify({"ok": True, "reply": f"🗓️ Zone {zone} set to {minutes} minutes."})

    return jsonify({"ok": False, "reply": f"🤔 I don’t recognize: “{cmd_raw}”"})


# Back-compat
@app.post("/command")
def command():
    return api_command()


@app.get("/api/schedule")
def api_get_schedule():
    return jsonify(load_schedule())


@app.post("/api/schedule/update")
def api_update_schedule():
    j = request.get_json(silent=True) or {}
    zone = str(j.get("zone", 1))
    minutes = _clamp_int(j.get("minutes", 10), default=10, lo=0, hi=240)

    data = load_schedule()
    data.setdefault("zones", {})
    data["zones"].setdefault(zone, {"minutes": 10, "enabled": True})
    data["zones"][zone]["minutes"] = minutes
    save_schedule(data)

    return jsonify({"ok": True, "zone": zone, "minutes": minutes})


@app.get("/api/schedule/plan")
def api_schedule_plan():
    if build_plan_for_today is None:
        return _err("schedule_manager not available (place schedule_manager.py).")

    try:
        cache = DATA / "hydration_cache.json"
        cache_data = _json_read(cache, {})
        score = float(cache_data.get("score", 5.0)) if isinstance(cache_data, dict) else 5.0
        plan = build_plan_for_today(score)
        return jsonify({"ok": True, "score": score, "plan": plan})
    except Exception as e:
        return _err(f"schedule_plan failed: {e}")


@app.post("/api/schedule/mark_ran")
def api_schedule_mark_ran():
    if mark_ran_today is None:
        return _err("schedule_manager not available (place schedule_manager.py).")

    try:
        mark_ran_today()
        return jsonify({"ok": True})
    except Exception as e:
        return _err(f"mark_ran failed: {e}")


@app.post("/api/health/eval")
def api_health_eval():
    if _evaluator is None:
        return _err("health_evaluator not available (place health_evaluator.py).")

    try:
        body = request.get_json(silent=True) or {}
        image_path = body.get("image_path") or str(DATA / "latest.jpg")
        res = _evaluator.evaluate_image(image_path)
        return jsonify({
            "ok": True,
            "greenness_score": res.greenness_score,
            "water_flag": res.water_flag,
            "dry_flag": res.dry_flag,
            "raw": res.raw
        })
    except Exception as e:
        return _err(f"health_eval failed: {e}")


@app.post("/api/hydration/score")
def api_hydration_score():
    if _hydrator is None or Inputs is None:
        return _err("hydration_engine not available (place hydration_engine.py).")

    try:
        b = request.get_json(silent=True) or {}
        inp = Inputs(
            soil_moisture_pct=b.get("soil_moisture_pct"),
            ambient_temp_f=b.get("ambient_temp_f"),
            humidity_pct=b.get("humidity_pct"),
            rain_24h_in=b.get("rain_24h_in", 0.0),
            rain_72h_in=b.get("rain_72h_in", 0.0),
            forecast_rain_24h_in=b.get("forecast_rain_24h_in", 0.0),
            greenness_score=b.get("greenness_score"),
            dry_flag=bool(b.get("dry_flag", False)),
            water_flag=bool(b.get("water_flag", False)),
        )
        result = _hydrator.compute(inp)
        return jsonify({
            "ok": True,
            "score": result.need_score,
            "advisory": result.advisory,
            "factors": result.factors
        })
    except Exception as e:
        return _err(f"hydration_score failed: {e}")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # Choose debug mode from env (safer than hardcoding)
    DEBUG = os.getenv("FLASK_DEBUG", "1").strip() in ("1", "true", "yes", "on")

    # Start overwatch only once (avoid double thread with Flask reloader)
    if (not DEBUG) or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        start_overwatch_thread_once()

    app.run(host="0.0.0.0", port=5051, debug=DEBUG)
