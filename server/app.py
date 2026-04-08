"""Entry point for openenv / uv_run deployment modes."""
from server.main import app  # noqa: F401 — re-exported for openenv


def main() -> None:
    import uvicorn
    uvicorn.run("server.main:app", host="0.0.0.0", port=7860, workers=1)


if __name__ == "__main__":
    main()
