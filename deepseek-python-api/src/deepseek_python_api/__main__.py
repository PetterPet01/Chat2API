from __future__ import annotations

from uvicorn import run

from .settings import get_settings


def main() -> None:
    settings = get_settings()
    run(
        "deepseek_python_api.app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
