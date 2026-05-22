from __future__ import annotations

import argparse

# ruff: noqa: D103
import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the ConvFinQA FastAPI app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run("convfinqa.serving.app:create_app", factory=True, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
