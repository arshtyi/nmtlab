import argparse

from nmtlab.config import CONFIG, RESULTS


def main() -> None:
    parser = argparse.ArgumentParser(prog="nmtlab")
    parser.add_argument("command", choices=["prepare", "app", "evaluate"])
    parser.add_argument("--device", choices=["cpu", "mps"], default=CONFIG["runtime"]["device"])
    args = parser.parse_args()
    if args.command == "prepare":
        from nmtlab.data import prepare

        prepare()
        print("Model and evaluation data are ready for offline use.")
        return
    from nmtlab.engine import Translator

    translator = Translator(args.device)
    if args.command == "evaluate":
        from nmtlab.evaluation import evaluate

        evaluate(translator, RESULTS, progress=lambda done, total: print(f"Evaluated {done}/{total} samples.", flush=True))
        print(f"Evaluation saved to {RESULTS}.")
    else:
        from nmtlab.ui import LAUNCH_OPTIONS, create_app

        translator.warmup()
        create_app(translator).queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", share=False, **LAUNCH_OPTIONS)


if __name__ == "__main__":
    main()
