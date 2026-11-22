"""
math_tools.py - safe bridge between Deep OQ and the two solver tools.

Put this file in the SAME folder as deep_oq.py, together with solve.py and bigcalc.py.

Why a subprocess: a hard integral or a huge factorial can take a long time or a lot of
memory. Running each job as a separate process with a time limit, a CPU limit and a small
concurrency cap means one bad question can never freeze or crash the web server.
Arguments are passed as a list (never through a shell), so user text cannot inject commands.
"""
import os
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
TIMEOUT = int(os.environ.get("MATH_TOOL_TIMEOUT", "25"))      # seconds per job
MAX_OUT = int(os.environ.get("MATH_TOOL_MAX_OUT", "6000"))    # characters returned
_slots = threading.Semaphore(int(os.environ.get("MATH_TOOL_SLOTS", "2")))

# which subcommands run in which script (whitelist: anything else is refused)
BIG = {"calc", "hpintegral", "hpsum", "hpidentify"}
FAST = {"integral", "sum", "limit", "roots", "opt", "identify", "minpoly", "identity", "search"}
# flags the planner may use (whitelist)
FLAGS = {"--lo", "--hi", "--var", "--vars", "--at", "--dir", "--alt", "--digits", "--bounds",
         "--maximize", "--min", "--max", "--collect", "--basis", "--maxcoef", "--show"}
_NO_VALUE = {"--alt", "--maximize"}


def _limits():
    try:
        import resource
        cpu = TIMEOUT + 5
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    except Exception:
        pass


def _run(script, args):
    cmd = [sys.executable, os.path.join(HERE, script)] + [str(a) for a in args]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    if not _slots.acquire(timeout=TIMEOUT):
        return None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT,
                           cwd=HERE, env=env, preexec_fn=_limits if os.name == "posix" else None)
    except subprocess.TimeoutExpired:
        return "TOOL TIMEOUT: the calculation took longer than %d seconds" % TIMEOUT
    except Exception as exc:
        print("math_tools error:", exc)
        return None
    finally:
        _slots.release()
    out = (r.stdout or "").strip()
    if r.returncode != 0 or not out:
        return None
    return out if len(out) <= MAX_OUT else out[:MAX_OUT] + "\n...(output cut)"


def _clean(spec):
    """spec = ["integral", "ln(1+x)/(1+x^2)", "0", "1", "--digits", "40"]. Returns a safe list or None."""
    if not isinstance(spec, (list, tuple)) or not spec:
        return None
    cmd = str(spec[0]).strip().lower()
    if cmd not in BIG and cmd not in FAST:
        return None
    args, i, rest = [cmd], 1, [str(x) for x in spec[1:]]
    while i <= len(rest):
        a = rest[i - 1]
        if len(a) > 400:
            return None
        if a.startswith("--"):
            if a not in FLAGS:
                return None
            args.append(a)
            if a not in _NO_VALUE:
                if i >= len(rest):
                    return None
                args.append(rest[i])
                i += 1
        else:
            args.append(a)
        i += 1
    return args


def run_math(spec):
    """Run one planned job. Returns the tool's text output, or None if refused/failed."""
    args = _clean(spec)
    if args is None:
        return None
    script = "bigcalc.py" if args[0] in BIG else "solve.py"
    return _run(script, args)


def big_calc(expr, digits=50):
    """Exact big-integer / high-precision arithmetic, e.g. big_calc('factorial(500)')."""
    expr = str(expr)
    if len(expr) > 400:
        return None
    out = _run("bigcalc.py", ["calc", expr, "--digits", int(digits)])
    return out


if __name__ == "__main__":      # quick check:  python math_tools.py
    tests = [
        ["calc", "2**200"],
        ["integral", "ln(1+x)/(1+x^2)", "0", "1"],
        ["hpintegral", "ln(x)^2/(1+x^2)", "0", "inf", "--digits", "40"],
        ["roots", "cos(x)-2*(1-cos(x))", "0", "pi"],
        ["rm", "-rf", "/"],                        # must be refused
        ["integral", "x", "0", "1", "--evil", "1"],  # must be refused
    ]
    for t in tests:
        res = run_math(t)
        print(">>", t)
        print(res if res is None else res.splitlines()[0:4])
        print()
