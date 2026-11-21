"""
chat_cli.py - try the Deep OQ engine from the terminal.

    python chat_cli.py --offline     solvers only: no API key, no internet, no cost
    python chat_cli.py               full engine with your API keys (3 panel agents by default)
    python chat_cli.py 7             full engine with 7 panel agents

Type a question and press Enter. Type  quit  to stop. Lines starting with [debug] are for you;
they show which solver, agents and checks ran. Users never see them.
"""
import sys


def offline() -> None:
    import solvers
    print(f"Offline mode: {len(solvers.REGISTRY)} solver modules, no AI models. Try: derivative of x^2*sin(x)")
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit"):
            break
        r = solvers.try_solve(q) if q else None
        if r is None:
            print("[no solver matched this question - the full engine would send it to the AI agents]")
            continue
        print(f"[{r.solver}] verified: {r.verified}\n{r.answer}\ncheck: {r.check}")
        for step in r.steps:
            print("  -", step)


def full() -> None:
    from providers import ProviderPool
    from reasoning_layer import build_engine

    pool = ProviderPool()
    if not pool.available():
        print("No API keys found. Set at least one first (see the steps I gave you), or use:  python chat_cli.py --offline")
        return
    print("Providers found:", ", ".join(pool.available()))
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3
    engine = build_engine(panel_agents=n, strong_agents=min(n, 2))
    history = []
    print(f"Engine ready with {n} panel agents. Type a message.")
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit"):
            break
        if not q:
            continue
        reply = engine.respond(history, q)
        print(f"\nDeep OQ> {reply.text}")
        print(f"[debug] mode={reply.mode} confidence={reply.confidence}")
        for note in reply.notes:
            print(f"[debug] {note}")
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": reply.text}]


if __name__ == "__main__":
    offline() if "--offline" in sys.argv else full()
