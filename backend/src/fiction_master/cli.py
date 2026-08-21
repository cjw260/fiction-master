from __future__ import annotations

import argparse

import uvicorn

from fiction_master.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(prog="fiction-master")
    parser.add_argument("command", choices=["serve"], nargs="?", default="serve")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    settings = get_settings()
    uvicorn.run(
        "fiction_master.main:app",
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=settings.app_env == "development",
    )


if __name__ == "__main__":
    main()
