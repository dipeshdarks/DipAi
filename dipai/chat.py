"""DIPAI terminal chat - an interactive REPL driving the real local model.

Run:
    python -m dipai.chat --ckpt checkpoints/scratch

Single-shot vision (image question answering, separate pretrained VLM):
    python -m dipai.chat --image path\\to\\photo.jpg --prompt "Describe this image"
    # (the text REPL stays the default; --image never touches the text model)

Commands (typed in the prompt):
    /new       start a fresh conversation (clears history)
    /status    show model + hardware status
    /settings  show / change generation settings
    /save      save the session transcript to a text file
    /clear     clear the screen
    /help      show help
    /exit      quit (Ctrl+C also works)

The model is a small scratch-trained LM: it continues text; it is not a
trained instruction assistant. Prompts phrased as plain text work best.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dipai.generate import GenerationError, GenerationSettings
from dipai.inference import ConversationSession, DIPAIEngine, EngineError
from dipai.utils import setup_logging

BANNER = r"""
============================================================
  DIPAI Chat - local text model (DIPDARKS)
  type /help for commands, /exit to quit
============================================================
"""

HELP_TEXT = """Commands:
  /new          start a new conversation
  /status       model + hardware status
  /settings     show current settings; use e.g.
                /settings temperature 0.7
                /settings top_k 50
                /settings top_p 0.9
                /settings max_new_tokens 300
                /settings repetition_penalty 1.1
  /save [file]  save transcript (default: chat_saved_<time>.txt)
  /clear        clear screen
  /help         this help
  /exit         quit
Anything else is sent to the model as a prompt."""

GENERATION_PARAMS = ("temperature", "top_k", "top_p", "max_new_tokens", "repetition_penalty")


def _print_settings(s: GenerationSettings) -> None:
    print(f"  settings: {s.describe()}")


def _handle_command(
    line: str,
    session: ConversationSession,
    settings: GenerationSettings,
    engine: DIPAIEngine,
) -> Optional[str]:
    """Handle a slash-command. Returns '/exit' to quit, None otherwise."""
    parts = line.split()
    cmd = parts[0].lower()

    if cmd in ("/exit", "/quit", "/q"):
        return "/exit"
    if cmd == "/help":
        print(HELP_TEXT)
    elif cmd == "/new":
        session.clear()
        print("  [new conversation started]")
    elif cmd == "/clear":
        print("\033[2J\033[H", end="")
    elif cmd == "/status":
        print(f"  {engine.status().line()}")
        print(f"  session turns: {len(session)}")
    elif cmd == "/settings":
        if len(parts) == 1:
            _print_settings(settings)
        elif len(parts) == 3:
            name, raw = parts[1], parts[2]
            if name not in GENERATION_PARAMS:
                print(f"  unknown setting '{name}'. Valid: {', '.join(GENERATION_PARAMS)}")
            else:
                try:
                    val = float(raw) if name != "top_k" and name != "max_new_tokens" else int(raw)
                    setattr(settings, name, val)
                    settings.validate()
                    _print_settings(settings)
                except (ValueError, GenerationError) as exc:
                    print(f"  invalid value: {exc}")
        else:
            print("  usage: /settings [name value]")
    elif cmd == "/save":
        out = Path(parts[1]) if len(parts) > 1 else Path(
            f"chat_saved_{datetime.now():%Y%m%d_%H%M%S}.txt"
        )
        try:
            with open(out, "w", encoding="utf-8") as f:
                f.write(f"# DIPAI chat transcript - {datetime.now():%Y-%m-%d %H:%M:%S}\n")
                f.write(f"# {engine.status().line()}\n\n")
                for t in session.turns:
                    f.write(f"{t['role']}:\n{t['content']}\n\n")
            print(f"  transcript saved: {out}")
        except OSError as exc:
            print(f"  could not save: {exc}")
    else:
        print(f"  unknown command {cmd} (try /help)")
    return None


def run_repl(engine: DIPAIEngine, settings: Optional[GenerationSettings] = None) -> int:
    session = ConversationSession()
    settings = settings or GenerationSettings()
    print(BANNER)
    print(f"  {engine.status().line()}\n")
    _print_settings(settings)
    print()

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[bye]")
            return 0

        if not line:
            continue
        if line.startswith("/"):
            result = _handle_command(line, session, settings, engine)
            if result == "/exit":
                print("[bye]")
                return 0
            continue

        # ---- generate ---------------------------------------------------
        stop_flag = threading.Event()

        def on_text(text: str, _prev: list = [""]) -> None:
            # Stream cumulative text: print only the delta.
            delta = text[len(_prev[0]):] if text.startswith(_prev[0]) else text
            _prev[0] = text
            print(delta, end="", flush=True)

        print("dipai> ", end="", flush=True)
        t0 = time.perf_counter()
        try:
            response = engine.generate(
                line, settings=settings, session=session,
                stream_fn=on_text, stop_event=stop_flag.is_set,
            )
            # If streaming already printed the text, just add newline;
            # if it printed nothing (tiny output), print full response.
            if response:
                print()
            elapsed = time.perf_counter() - t0
            print(f"        [{len(response)} chars, {elapsed:.1f}s]\n")
        except GenerationError as exc:
            print(f"\n[error: {exc}]\n")
        except KeyboardInterrupt:
            stop_flag.set()
            print("\n[stopped]\n")
        except Exception as exc:  # noqa: BLE001 - report, keep REPL alive
            print(f"\n[generation failed: {exc}]\n")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="DIPAI terminal chat",
        epilog="Vision example: python -m dipai.chat --image photo.jpg "
               "--prompt \"What is in this picture?\"",
    )
    ap.add_argument("--ckpt", "-k", default="checkpoints/scratch",
                    help="scratch checkpoint directory (default: checkpoints/scratch)")
    ap.add_argument("--adapter", "-a", default=None,
                    help="PEFT adapter directory from dipai.finetune (Mode B); "
                         "overrides --ckpt")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--once", metavar="PROMPT", default=None,
                    help="single-shot: answer PROMPT and exit (non-interactive)")
    ap.add_argument("--image", metavar="PATH", default=None,
                    help="single-shot VISION mode: answer --prompt about PATH "
                         "using the separate vision model, then exit "
                         "(the text model is not loaded)")
    ap.add_argument("--prompt", metavar="TEXT", default=None,
                    help="question for --image mode (default: describe the image)")
    ap.add_argument("--vision-config", default="configs/vision.yaml",
                    help="vision YAML config (default: configs/vision.yaml; "
                         "missing file = built-in defaults)")
    return ap


def _run_vision(args: argparse.Namespace, log) -> int:
    """--image single-shot path: loads ONLY the vision model."""
    from dipai import vision
    from dipai.config import load_vision_config
    from dipai.vision import VisionError

    try:
        cfg = load_vision_config(args.vision_config, required=False)
        vision.load_vision_model(cfg)
        answer = vision.answer_about_image(
            args.image, args.prompt or vision.DEFAULT_DESCRIBE_PROMPT
        )
    except VisionError as exc:
        log.error("vision failed: %s", exc)
        return 1
    print(answer)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    log = setup_logging()
    args = build_arg_parser().parse_args(argv)

    if args.image is not None:
        # Vision single-shot: skip the text engine entirely so this works
        # even when no text checkpoint exists.
        return _run_vision(args, log)

    try:
        if args.adapter:
            engine = DIPAIEngine.from_adapter(args.adapter, device=args.device)
        else:
            engine = DIPAIEngine.from_checkpoint(args.ckpt, device=args.device)
    except EngineError as exc:
        log.error("cannot start chat: %s", exc)
        return 2

    settings = GenerationSettings(
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )

    if args.once is not None:
        try:
            response = engine.generate(args.once, settings=settings)
        except Exception as exc:  # noqa: BLE001
            log.error("generation failed: %s", exc)
            return 1
        print(response)
        return 0

    return run_repl(engine, settings)


if __name__ == "__main__":
    raise SystemExit(main())
