"""
providers.py - one interface to Anthropic, OpenAI, Gemini and Groq (plain HTTPS, no SDKs).

Set whichever API keys you have as environment variables (on Render: Environment tab):

    ANTHROPIC_API_KEY   OPENAI_API_KEY   GEMINI_API_KEY (or GOOGLE_API_KEY)   GROQ_API_KEY

Optional extra providers (OpenAI-compatible; each needs its key AND a model name):
    OPENROUTER_API_KEY + OPENROUTER_MODEL   (one OpenRouter key reaches hundreds of models)
    MISTRAL, XAI, DEEPSEEK, TOGETHER, FIREWORKS, CEREBRAS, PERPLEXITY, SAMBANOVA, NVIDIA:
    <NAME>_API_KEY + <NAME>_MODEL (and optionally <NAME>_FAST_MODEL)

Optional tuning (no code changes needed):
    DEEP_OQ_TIMEOUT=900      seconds to wait for one model reply
    DEEP_OQ_TOKEN_CAP=64000  ceiling on reply length; if a model allows less, the call halves it and retries

Optional overrides:
    ANTHROPIC_MODEL / OPENAI_MODEL / GEMINI_MODEL / GROQ_MODEL   pin an exact model ID
    DEEP_OQ_LEAD=anthropic,openai,gemini,groq                    order tried for the lead model

Model names change often, so each provider has an ordered preference list (strongest
first). At first use the provider asks its own "list models" endpoint and takes the first
preference your key can actually use. If a call fails because a model is unknown, it
moves to the next one. Keys are only ever sent to the provider's own host, are never
logged and never put in URLs.

    python providers.py --check      # shows which keys are set and which model each resolves to
    python providers.py --selftest   # offline tests (no keys, no network)
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Callable, Dict, List, Optional, Tuple


class ProviderError(Exception):
    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def http_json(method: str, url: str, headers: Optional[dict] = None, body: Optional[dict] = None,
              timeout: Optional[float] = None, retries: int = 3) -> dict:
    timeout = timeout or _env_int("DEEP_OQ_TIMEOUT", 900)   # seconds; long answers from thinking models need time
    data = json.dumps(body).encode("utf-8") if body is not None else None
    hdrs = {"content-type": "application/json", **(headers or {})}
    last: Optional[ProviderError] = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", "replace")[:1500]
            last = ProviderError(f"HTTP {e.code}: {text}", e.code)
            if e.code in (429, 500, 502, 503, 504, 529) and attempt < retries - 1:
                time.sleep(min(1.5 * 2 ** attempt, 20))
                continue
            raise last
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = ProviderError(f"network error: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise last
    raise last or ProviderError("request failed")


# --------------------------------------------------------------------------
# Message helpers
# --------------------------------------------------------------------------


def to_turns(messages: List[dict]) -> Tuple[str, List[dict]]:
    """Split out system text; merge same-role neighbours; make sure we start with a user turn.
    A message may carry "images": [{"media_type": ..., "data": <base64>}]; they stay on their turn."""
    system_parts: List[str] = []
    turns: List[dict] = []
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        imgs = m.get("images") or []
        if not content and not imgs:
            continue
        if role == "system":
            system_parts.append(content)
            continue
        if role not in ("user", "assistant"):
            continue
        if imgs and not content:
            content = "(see the attached image)"
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n\n" + content
            if imgs:
                turns[-1].setdefault("images", []).extend(imgs)
        else:
            turn = {"role": role, "content": content}
            if imgs:
                turn["images"] = list(imgs)
            turns.append(turn)
    if not turns or turns[0]["role"] != "user":
        turns.insert(0, {"role": "user", "content": "(conversation start)"})
    return "\n\n".join(system_parts), turns


_IMG_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
MAX_IMAGE_B64 = 7_000_000  # about 5 MB of picture data, the usual provider limit


def normalize_images(images) -> List[dict]:
    """Accept data-URL strings or {"media_type", "data"} dicts (data = base64 text or raw bytes)."""
    out: List[dict] = []
    for im in images or []:
        if isinstance(im, str):
            m = re.match(r"data:(image/[a-z+.\-]+);base64,(.+)$", im.strip(), re.S)
            if not m:
                raise ValueError("the image is not in a format I can read")
            mt, data = m.group(1), m.group(2)
        else:
            mt = im.get("media_type") or im.get("mime_type") or ""
            data = im.get("data", "")
            if isinstance(data, bytes):
                data = base64.b64encode(data).decode("ascii")
        mt = "image/jpeg" if mt == "image/jpg" else mt
        if mt not in _IMG_TYPES:
            raise ValueError("only PNG, JPEG, GIF and WebP pictures are supported")
        if not data:
            raise ValueError("the image is empty")
        if len(data) > MAX_IMAGE_B64:
            raise ValueError("the image is too large (about 5 MB is the limit)")
        out.append({"media_type": mt, "data": data})
    return out


def _clamp_effort(effort: Optional[str], allowed: Tuple[str, ...]) -> Optional[str]:
    if not effort:
        return None
    order = ["low", "medium", "high", "xhigh", "max"]
    if effort in allowed:
        return effort
    if effort in order:
        # nearest allowed level at or below, else the lowest allowed
        for lvl in reversed(order[: order.index(effort) + 1]):
            if lvl in allowed:
                return lvl
    return allowed[0] if allowed else None


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------


class Provider:
    name = ""
    env_keys: Tuple[str, ...] = ()
    env_model = ""
    strong: List[str] = []
    fast: List[str] = []
    vision = False          # can this provider read pictures?
    token_floor = 8000      # reasoning models spend max_tokens on thinking too
    token_cap = 64000
    default_effort = "high"

    def __init__(self, api_key: str, tier: str = "strong", model: Optional[str] = None):
        # `model` pins an exact model ID. ProviderPool passes it from your environment variables.
        self.api_key = api_key
        self.tier = tier
        self.override = model
        self._available: Optional[List[str]] = None
        self._good: Optional[str] = None

    # -- to implement per provider
    def list_models(self) -> List[str]:
        raise NotImplementedError

    def _request(self, model: str, messages: List[dict], max_tokens: int,
                 effort: Optional[str], minimal: bool) -> Tuple[str, str]:
        raise NotImplementedError

    # -- model selection
    def preferences(self) -> List[str]:
        prefs = list(self.strong if self.tier == "strong" else self.fast)
        if self.override:
            prefs = [self.override] + [p for p in prefs if p != self.override]
        return prefs

    def ordered_models(self) -> List[str]:
        if self._good:
            return [self._good]
        prefs = self.preferences()
        if self.override:
            return prefs
        if self._available is None:
            try:
                self._available = self.list_models()
            except Exception:  # noqa: BLE001
                self._available = []
        have = set(self._available)
        first = [p for p in prefs if p in have]
        return first + [p for p in prefs if p not in have] if first else prefs

    # -- public call
    def call(self, messages: List[dict], **opts) -> dict:
        cap = _env_int("DEEP_OQ_TOKEN_CAP", self.token_cap)      # raise this if your models allow more
        want = int(opts.get("max_tokens") or 0)
        max_tokens = min(max(want, self.token_floor), cap)
        effort = opts.get("effort") or self.default_effort
        last: Optional[Exception] = None
        for model in self.ordered_models():
            minimal = False
            for attempt in range(8):
                try:
                    text, finish = self._request(model, messages, max_tokens, effort, minimal)
                    if not text.strip() and finish == "length" and max_tokens < cap:
                        max_tokens = min(cap, max_tokens * 2)
                        text, finish = self._request(model, messages, max_tokens, effort, minimal)
                    self._good = model
                    return {"text": text, "finish_reason": finish, "model": model, "provider": self.name}
                except ProviderError as e:
                    last = e
                    msg = str(e).lower()
                    if e.status == 400 and max_tokens > self.token_floor // 2 and any(
                            k in msg for k in ("max_tokens", "max_completion_tokens", "maxoutputtokens",
                                               "max_output_tokens", "output tokens", "too large", "exceeds")):
                        max_tokens = max(self.token_floor // 2, max_tokens // 2)   # model allows less: halve and retry
                        continue
                    if e.status == 400 and not minimal and any(
                            k in msg for k in ("effort", "reasoning", "output_config", "thinking",
                                               "temperature", "unsupported", "unknown field", "unknown name")):
                        minimal = True      # retry once without optional tuning fields
                        continue
                    break
            if isinstance(last, ProviderError) and last.status in (400, 404) and "model" in str(last).lower():
                continue                    # unknown / unavailable model: try next preference
            if isinstance(last, ProviderError) and last.status in (401, 403):
                raise last                  # bad key: other models won't help
            if isinstance(last, ProviderError) and last.status in (429, 500, 502, 503, 529, None):
                raise last
        raise last or ProviderError(f"{self.name}: no usable model")


class AnthropicProvider(Provider):
    name = "anthropic"
    vision = True
    env_keys = ("ANTHROPIC_API_KEY",)
    env_model = "ANTHROPIC_MODEL"
    strong = ["claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5"]
    fast = ["claude-sonnet-5-5", "claude-haiku-5-5", "claude-sonnet-5"]
    base = "https://api.anthropic.com/v1"

    def _headers(self) -> dict:
        return {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}

    def list_models(self) -> List[str]:
        out = http_json("GET", f"{self.base}/models?limit=100", self._headers(), timeout=30, retries=2)
        return [m["id"] for m in out.get("data", [])]

    def _request(self, model, messages, max_tokens, effort, minimal):
        system, turns = to_turns(messages)
        body: dict = {"model": model, "max_tokens": max_tokens, "messages": [self._turn(t) for t in turns]}
        if system:
            body["system"] = system
        # temperature / top_p / top_k are deliberately never sent: current Claude models reject them.
        if effort and not minimal:
            body["output_config"] = {"effort": effort}
        return self.parse(http_json("POST", f"{self.base}/messages", self._headers(), body))

    @staticmethod
    def _turn(t: dict) -> dict:
        if not t.get("images"):
            return {"role": t["role"], "content": t["content"]}
        blocks = [{"type": "image", "source": {"type": "base64", "media_type": i["media_type"], "data": i["data"]}}
                  for i in t["images"]]
        blocks.append({"type": "text", "text": t["content"]})
        return {"role": t["role"], "content": blocks}

    @staticmethod
    def parse(resp: dict) -> Tuple[str, str]:
        text = "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")
        stop = resp.get("stop_reason")
        finish = {"max_tokens": "length", "end_turn": "stop", "stop_sequence": "stop",
                  "refusal": "refusal"}.get(stop, stop or "stop")
        return text, finish


class OpenAIProvider(Provider):
    name = "openai"
    vision = True
    env_keys = ("OPENAI_API_KEY",)
    env_model = "OPENAI_MODEL"
    strong = ["gpt-6-astra", "gpt-5.6-sol", "gpt-5.5"]
    fast = ["gpt-6.1-sol", "gpt-6.1-terra", "gpt-5.6-terra", "gpt-6-luna"]
    base = "https://api.openai.com/v1"

    def _headers(self) -> dict:
        return {"authorization": f"Bearer {self.api_key}"}

    def list_models(self) -> List[str]:
        out = http_json("GET", f"{self.base}/models", self._headers(), timeout=30, retries=2)
        return [m["id"] for m in out.get("data", [])]

    def _request(self, model, messages, max_tokens, effort, minimal):
        system, turns = to_turns(messages)
        body: dict = {"model": model, "input": [self._turn(t) for t in turns], "max_output_tokens": max_tokens}
        if system:
            body["instructions"] = system
        if effort and not minimal:
            body["reasoning"] = {"effort": effort}
        return self.parse(http_json("POST", f"{self.base}/responses", self._headers(), body))

    @staticmethod
    def _turn(t: dict) -> dict:
        if not t.get("images"):
            return {"role": t["role"], "content": t["content"]}
        blocks = [{"type": "input_image", "image_url": f"data:{i['media_type']};base64,{i['data']}"} for i in t["images"]]
        blocks.append({"type": "input_text", "text": t["content"]})
        return {"role": t["role"], "content": blocks}

    @staticmethod
    def parse(resp: dict) -> Tuple[str, str]:
        parts: List[str] = []
        for item in resp.get("output", []):
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        parts.append(c.get("text", ""))
                    elif c.get("type") == "refusal":
                        parts.append(c.get("refusal", ""))
        finish = "stop"
        if resp.get("status") == "incomplete":
            reason = (resp.get("incomplete_details") or {}).get("reason", "")
            finish = "length" if "max" in reason else ("refusal" if "filter" in reason else "stop")
        return "".join(parts), finish


class GroqProvider(Provider):
    name = "groq"
    env_keys = ("GROQ_API_KEY",)
    env_model = "GROQ_MODEL"
    strong = ["qwen/qwen3.6-27b", "openai/gpt-oss-120b", "llama-3.3-70b-versatile"]
    fast = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.1-8b-instant"]
    base = "https://api.groq.com/openai/v1"
    token_floor = 4000
    token_cap = 30000

    def _headers(self) -> dict:
        return {"authorization": f"Bearer {self.api_key}"}

    def list_models(self) -> List[str]:
        out = http_json("GET", f"{self.base}/models", self._headers(), timeout=30, retries=2)
        return [m["id"] for m in out.get("data", [])]

    def _request(self, model, messages, max_tokens, effort, minimal):
        system, turns = to_turns(messages)
        msgs = ([{"role": "system", "content": system}] if system else []) + turns
        body: dict = {"model": model, "messages": msgs, "max_completion_tokens": max_tokens}
        if model.startswith("openai/gpt-oss") and effort and not minimal:
            body["reasoning_effort"] = _clamp_effort(effort, ("low", "medium", "high"))
        return self.parse(http_json("POST", f"{self.base}/chat/completions", self._headers(), body))

    @staticmethod
    def parse(resp: dict) -> Tuple[str, str]:
        ch = (resp.get("choices") or [{}])[0]
        text = (ch.get("message") or {}).get("content") or ""
        finish = {"length": "length", "stop": "stop", "content_filter": "refusal"}.get(
            ch.get("finish_reason"), ch.get("finish_reason") or "stop")
        return text, finish


class GeminiProvider(Provider):
    name = "gemini"
    vision = True
    env_keys = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    env_model = "GEMINI_MODEL"
    strong = ["gemini-4-argon", "gemini-4-carbon", "gemini-3.8-flash", "gemini-3.1-pro-preview"]
    fast = ["gemini-3.8-flash", "gemini-3.7-flash"]
    base = "https://generativelanguage.googleapis.com/v1beta"
    token_cap = 32768      # non-streaming limit for one response

    def _headers(self) -> dict:
        return {"x-goog-api-key": self.api_key}

    def list_models(self) -> List[str]:
        out = http_json("GET", f"{self.base}/models?pageSize=200", self._headers(), timeout=30, retries=2)
        return [m["name"].split("/", 1)[-1] for m in out.get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])]

    def _request(self, model, messages, max_tokens, effort, minimal):
        system, turns = to_turns(messages)
        body: dict = {
            "contents": [{"role": "user" if t["role"] == "user" else "model",
                          "parts": [{"inline_data": {"mime_type": i["media_type"], "data": i["data"]}}
                                    for i in t.get("images", [])] + [{"text": t["content"]}]} for t in turns],
            "generationConfig": {"maxOutputTokens": max_tokens},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return self.parse(http_json("POST", f"{self.base}/models/{model}:generateContent",
                                    self._headers(), body))

    @staticmethod
    def parse(resp: dict) -> Tuple[str, str]:
        cands = resp.get("candidates") or []
        if not cands:
            return "", "refusal" if resp.get("promptFeedback", {}).get("blockReason") else "stop"
        c = cands[0]
        text = "".join(p.get("text", "") for p in (c.get("content") or {}).get("parts", [])
                       if not p.get("thought"))
        finish = {"MAX_TOKENS": "length", "STOP": "stop", "SAFETY": "refusal",
                  "RECITATION": "refusal"}.get(c.get("finishReason"), "stop")
        return text, finish


PROVIDER_CLASSES = [AnthropicProvider, OpenAIProvider, GeminiProvider, GroqProvider]


class OpenAICompatProvider(Provider):
    """Any service that speaks OpenAI's /chat/completions format. There is no built-in model
    guess for these: set <NAME>_MODEL (and optionally <NAME>_FAST_MODEL) or the provider stays off."""
    base = ""
    token_floor = 4000
    token_cap = 30000

    def _headers(self) -> dict:
        return {"authorization": f"Bearer {self.api_key}"}

    def list_models(self) -> List[str]:
        out = http_json("GET", f"{self.base}/models", self._headers(), timeout=30, retries=2)
        return [m["id"] for m in out.get("data", [])]

    def _request(self, model, messages, max_tokens, effort, minimal):
        system, turns = to_turns(messages)
        msgs = ([{"role": "system", "content": system}] if system else []) + turns
        body: dict = {"model": model, "messages": msgs, "max_tokens": max_tokens}
        return GroqProvider.parse(http_json("POST", f"{self.base}/chat/completions", self._headers(), body))


# name, API-key variable, base URL. All of these are optional extras.
COMPAT_SPECS = [
    ("openrouter", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),   # one key, hundreds of models
    ("mistral", "MISTRAL_API_KEY", "https://api.mistral.ai/v1"),
    ("xai", "XAI_API_KEY", "https://api.x.ai/v1"),
    ("deepseek", "DEEPSEEK_API_KEY", "https://api.deepseek.com/v1"),
    ("together", "TOGETHER_API_KEY", "https://api.together.xyz/v1"),
    ("fireworks", "FIREWORKS_API_KEY", "https://api.fireworks.ai/inference/v1"),
    ("cerebras", "CEREBRAS_API_KEY", "https://api.cerebras.ai/v1"),
    ("perplexity", "PERPLEXITY_API_KEY", "https://api.perplexity.ai"),
    ("sambanova", "SAMBANOVA_API_KEY", "https://api.sambanova.ai/v1"),
    ("nvidia", "NVIDIA_API_KEY", "https://integrate.api.nvidia.com/v1"),
]


def _make_compat(name: str, env_key: str, base: str):
    return type(f"{name.title()}Provider", (OpenAICompatProvider,),
                {"name": name, "env_keys": (env_key,), "env_model": f"{name.upper()}_MODEL", "base": base})


COMPAT_CLASSES = [_make_compat(*spec) for spec in COMPAT_SPECS]
ALL_CLASSES = PROVIDER_CLASSES + COMPAT_CLASSES


# --------------------------------------------------------------------------
# Pool: lead model with failover + a panel of agents spread over providers
# --------------------------------------------------------------------------


class ProviderPool:
    def __init__(self, env: Optional[dict] = None):
        env = env if env is not None else os.environ
        self.strong: Dict[str, Provider] = {}
        self.fast: Dict[str, Provider] = {}
        for cls in ALL_CLASSES:
            key = next((env[k].strip() for k in cls.env_keys if env.get(k, "").strip()), "")
            if not key:
                continue
            pinned = env.get(cls.env_model, "").strip() or None
            fast_pinned = env.get(f"{cls.name.upper()}_FAST_MODEL", "").strip() or None
            if issubclass(cls, OpenAICompatProvider) and not pinned:
                continue                      # extras need an explicit model name
            self.strong[cls.name] = cls(key, "strong", model=pinned)
            self.fast[cls.name] = cls(key, "fast", model=fast_pinned or (pinned if issubclass(cls, OpenAICompatProvider) else None))
        order = [x.strip() for x in env.get("DEEP_OQ_LEAD", "").split(",") if x.strip()]
        default = [c.name for c in ALL_CLASSES]
        self.order = [n for n in order + default if n in self.strong]
        self.order = list(dict.fromkeys(self.order))

    def available(self) -> List[str]:
        return list(self.order)

    def lead_call(self) -> Callable:
        """Most powerful model first; if that provider errors, the next one answers."""
        if not self.order:
            raise ProviderError("No API keys found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, "
                                "GEMINI_API_KEY or GROQ_API_KEY.")

        def call(messages, **opts):
            last: Optional[Exception] = None
            for name in self.order:
                try:
                    return self.strong[name].call(messages, **opts)
                except ProviderError as e:
                    last = e
            raise last or ProviderError("all providers failed")
        return call

    def vision_call(self) -> Optional[Callable]:
        """Lead call restricted to providers that can read pictures (Anthropic, OpenAI, Gemini).
        Returns None when none of your keys belongs to such a provider."""
        names = [n for n in self.order if self.strong[n].vision]
        if not names:
            return None

        def call(messages, **opts):
            last: Optional[Exception] = None
            for name in names:
                try:
                    return self.strong[name].call(messages, **opts)
                except ProviderError as e:
                    last = e
            raise last or ProviderError("all image-capable providers failed")
        return call

    def agents(self, n: int, strong_agents: int = 4) -> List[Tuple[Callable, int, str]]:
        """n panel agents, round-robin over providers. The first `strong_agents` use each
        provider's strongest model (weight 2), the rest its fast tier (weight 1)."""
        out: List[Tuple[Callable, int, str]] = []
        names = self.order
        for i in range(n):
            name = names[i % len(names)]
            strong = i < strong_agents
            prov = (self.strong if strong else self.fast)[name]
            out.append((prov.call, 2 if strong else 1, f"{name}/{'strong' if strong else 'fast'}"))
        return out

    def status(self) -> List[dict]:
        rows = []
        for name in self.order:
            row = {"provider": name}
            for tier, group in (("strong", self.strong), ("fast", self.fast)):
                p = group[name]
                try:
                    row[tier] = p.ordered_models()[0]
                except Exception as e:  # noqa: BLE001
                    row[tier] = f"error: {e}"
            rows.append(row)
        return rows


# --------------------------------------------------------------------------
# CLI + offline self-test
# --------------------------------------------------------------------------


def _selftest() -> None:
    system, turns = to_turns([
        {"role": "system", "content": "S1"}, {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "a"}, {"role": "user", "content": "b"},
        {"role": "assistant", "content": ""}])
    assert system == "S1" and turns[0]["role"] == "user"
    assert [t["role"] for t in turns] == ["user", "assistant", "user"] or turns[1]["role"] == "assistant"
    assert "a\n\nb" in turns[-1]["content"], turns

    # response parsers
    assert AnthropicProvider.parse({"content": [{"type": "thinking", "thinking": "x"},
                                                {"type": "text", "text": "Hi"}],
                                    "stop_reason": "max_tokens"}) == ("Hi", "length")
    assert OpenAIProvider.parse({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
                                 "output": [{"type": "message",
                                             "content": [{"type": "output_text", "text": "Yo"}]}]}) == ("Yo", "length")
    assert GroqProvider.parse({"choices": [{"message": {"content": "G"}, "finish_reason": "stop"}]}) == ("G", "stop")
    assert GeminiProvider.parse({"candidates": [{"content": {"parts": [{"text": "t", "thought": True},
                                                                       {"text": "Ans"}]},
                                                 "finishReason": "MAX_TOKENS"}]}) == ("Ans", "length")
    assert _clamp_effort("max", ("low", "medium", "high")) == "high"

    # request shaping + failover with a fake network
    sent: List[Tuple[str, dict]] = []

    def fake_http(method, url, headers=None, body=None, timeout=300, retries=3):
        sent.append((url, body or {}))
        if url.endswith("/models?limit=100"):
            return {"data": [{"id": "claude-opus-5-5"}]}
        if body and body.get("model") == "claude-fable-5-2":
            raise ProviderError("HTTP 404: model not found", 404)
        if body and "output_config" in body and body["model"] == "claude-opus-5-5" and not getattr(fake_http, "ok", False):
            fake_http.ok = True
            raise ProviderError("HTTP 400: unknown field output_config", 400)
        return {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}

    global http_json
    real = http_json
    http_json = fake_http
    try:
        p = AnthropicProvider("sk-test", "strong")
        r = p.call([{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}],
                   max_tokens=100, temperature=0.9)
        assert r["text"] == "ok" and r["model"] == "claude-opus-5-5", r
        posts = [b for u, b in sent if u.endswith("/messages")]
        assert all("temperature" not in b for b in posts), "temperature must never be sent"
        assert posts[0]["max_tokens"] == 8000 and "output_config" in posts[0]
        assert "output_config" not in posts[-1], "optional field dropped on 400 retry"
    finally:
        http_json = real

    pool = ProviderPool({"GROQ_API_KEY": "g", "ANTHROPIC_API_KEY": "a", "DEEP_OQ_LEAD": "groq"})
    assert pool.available() == ["groq", "anthropic"]
    ag = pool.agents(6, strong_agents=3)
    assert len(ag) == 6 and ag[0][1] == 2 and ag[5][1] == 1
    assert ProviderPool({}).available() == []

    # optional OpenAI-compatible extras: need a key AND an explicit model
    p2 = ProviderPool({"OPENROUTER_API_KEY": "k", "MISTRAL_API_KEY": "k2", "OPENROUTER_MODEL": "some/model"})
    assert p2.available() == ["openrouter"], p2.available()
    assert p2.agents(3)[0][2].startswith("openrouter")
    assert len(ALL_CLASSES) >= 14
    pinned = ProviderPool({"ANTHROPIC_API_KEY": "a", "ANTHROPIC_MODEL": "claude-opus-5-5"})
    assert pinned.strong["anthropic"].ordered_models()[0] == "claude-opus-5-5"

    captured: List[Tuple[str, dict]] = []

    def fake_http2(method, url, headers=None, body=None, timeout=300, retries=3):
        captured.append((url, body or {}))
        return {"choices": [{"message": {"content": "hi"}, "finish_reason": "length"}]}

    http_json = fake_http2
    try:
        out = p2.strong["openrouter"].call([{"role": "system", "content": "s"}, {"role": "user", "content": "q"}], max_tokens=10)
    finally:
        http_json = real
    assert out["text"] == "hi" and out["finish_reason"] == "length" and out["model"] == "some/model"
    assert captured[-1][0] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured[-1][1]["messages"][0]["role"] == "system" and captured[-1][1]["max_tokens"] == 4000
    # ---- images
    img = {"media_type": "image/png", "data": "QUJD"}
    _, tt = to_turns([{"role": "user", "content": "prove this", "images": [img]}, {"role": "user", "content": "please"}])
    assert tt[0]["images"] == [img] and "please" in tt[0]["content"]
    a = AnthropicProvider._turn(tt[0])
    assert a["content"][0]["type"] == "image" and a["content"][0]["source"]["data"] == "QUJD" and a["content"][-1]["type"] == "text"
    o = OpenAIProvider._turn(tt[0])
    assert o["content"][0] == {"type": "input_image", "image_url": "data:image/png;base64,QUJD"}
    sent_g: List[dict] = []

    def fake_g(method, url, headers=None, body=None, timeout=300, retries=3):
        sent_g.append(body or {})
        return {"candidates": [{"content": {"parts": [{"text": "seen"}]}, "finishReason": "STOP"}]}

    http_json = fake_g
    try:
        gp = GeminiProvider("k", "strong", model="gemini-test")
        assert gp.call([{"role": "user", "content": "what is this", "images": [img]}])["text"] == "seen"
    finally:
        http_json = real
    assert sent_g[-1]["contents"][0]["parts"][0] == {"inline_data": {"mime_type": "image/png", "data": "QUJD"}}
    assert normalize_images(["data:image/jpg;base64,QUJD"])[0]["media_type"] == "image/jpeg"
    assert normalize_images([{"media_type": "image/webp", "data": b"abc"}])[0]["data"] == "YWJj"
    for bad in (["not an image"], [{"media_type": "image/bmp", "data": "QUJD"}], [{"media_type": "image/png", "data": ""}]):
        try:
            normalize_images(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    assert ProviderPool({"GROQ_API_KEY": "g"}).vision_call() is None
    assert ProviderPool({"GROQ_API_KEY": "g", "OPENAI_API_KEY": "o"}).vision_call() is not None
    # ---- limits: a higher token ceiling is honoured, and a model that allows less is handled by halving
    seen_tokens: List[int] = []

    def fake_limit(method, url, headers=None, body=None, timeout=None, retries=3):
        seen_tokens.append(body["max_tokens"])
        if body["max_tokens"] > 40000:
            raise ProviderError("HTTP 400: max_tokens: 90000 > 32000, which is the maximum allowed", 400)
        return {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}

    http_json = fake_limit
    os.environ["DEEP_OQ_TOKEN_CAP"] = "100000"
    try:
        r = AnthropicProvider("k", "strong", model="claude-test").call([{"role": "user", "content": "q"}], max_tokens=90000)
    finally:
        http_json = real
        os.environ.pop("DEEP_OQ_TOKEN_CAP", None)
    assert r["text"] == "ok" and seen_tokens == [90000, 45000, 22500], seen_tokens
    assert _env_int("DEEP_OQ_TIMEOUT_NOT_SET", 900) == 900
    print(f"providers selftest passed ({len(ALL_CLASSES)} providers)")


def _check() -> None:
    pool = ProviderPool()
    if not pool.available():
        print("No API keys found in the environment.")
        return
    for row in pool.status():
        print(row)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    elif "--check" in sys.argv:
        _check()
    else:
        print(__doc__)
