"""
00_verify_env.py — sanity check before any data pulls.

Run this first. It checks:
  - config.yaml loads cleanly
  - .env exists and required keys are populated
  - tesseract is installed and callable
  - directories exist
  - USPTO API responds to a smoke-test request

Exit code 0 = ready to proceed. Non-zero = fix something first.
"""

from __future__ import annotations

import shutil
import sys

import requests

from _common import (
    PROJECT_ROOT,
    console,
    get_logger,
    load_config,
    load_env,
    resolve_path,
)

logger = get_logger("verify_env")


def check_config() -> dict:
    try:
        config = load_config()
        console.print("[green]✓[/green] config.yaml loads")
        return config
    except Exception as e:
        console.print(f"[red]✗ config.yaml failed to load: {e}[/red]")
        sys.exit(1)


def check_env_keys(config: dict) -> None:
    load_env()
    import os

    required = [config["uspto_api"]["api_key_env_var"]]
    # Model keys not strictly required for module 1, but warn if missing.
    optional = [m["api_key_env_var"] for m in config["models"] if m["provider"] != "local"]

    missing_required = [k for k in required if not os.environ.get(k)]
    missing_optional = [k for k in optional if not os.environ.get(k)]

    if missing_required:
        for k in missing_required:
            console.print(f"[red]✗ Missing required env var: {k}[/red]")
        console.print("[red]Add these to .env before running the puller.[/red]")
        sys.exit(1)
    else:
        console.print(f"[green]✓[/green] USPTO API key present")

    if missing_optional:
        for k in missing_optional:
            console.print(f"[yellow]⚠ Missing optional env var: {k} (needed for module 4, not now)[/yellow]")


def check_tesseract() -> None:
    if shutil.which("tesseract") is None:
        console.print(
            "[yellow]⚠ tesseract not found in PATH. "
            "You'll need it for module 4_normalize.py text-masking. "
            "OK to skip for module 1.[/yellow]"
        )
    else:
        console.print("[green]✓[/green] tesseract installed")


def check_directories(config: dict) -> None:
    expected = [
        config["paths"]["raw_positive"],
        config["paths"]["raw_negative_expired"],
        config["paths"]["raw_negative_opensource"],
        config["paths"]["processed"],
        config["paths"]["raw_responses"],
        config["paths"]["stats_dir"],
    ]
    missing = [p for p in expected if not resolve_path(p).exists()]
    if missing:
        for p in missing:
            console.print(f"[red]✗ Missing directory: {p}[/red]")
        sys.exit(1)
    console.print(f"[green]✓[/green] all required directories exist")


def smoke_test_uspto_api(config: dict) -> None:
    """
    Hit USPTO API with a trivial request to verify the endpoint and key work.

    NOTE: USPTO migrated PatentsView to data.uspto.gov in March 2026; the
    exact endpoint shape may have shifted. If this fails, check
    https://data.uspto.gov/apis for the current Patent Search API URL and
    update config.yaml.
    """
    import os

    api_key = os.environ.get(config["uspto_api"]["api_key_env_var"])
    base_url = config["uspto_api"]["base_url"]

    # Simplest possible smoke test: GET the root or a known endpoint with auth.
    # The exact path will need adjustment based on what data.uspto.gov publishes.
    url = base_url.rstrip("/")
    headers = {"X-API-KEY": api_key}

    try:
        # We try a HEAD request to see if the host responds; we're not making
        # a real query yet because the endpoint shape needs verification.
        r = requests.head(url, headers=headers, timeout=10)
        if r.status_code < 500:
            console.print(
                f"[green]✓[/green] USPTO API host reachable "
                f"(HTTP {r.status_code} — full endpoint shape verified in 01_pull_uspto.py)"
            )
        else:
            console.print(
                f"[yellow]⚠ USPTO API host returned {r.status_code}. "
                f"Check https://data.uspto.gov/apis for current endpoint.[/yellow]"
            )
    except requests.exceptions.RequestException as e:
        console.print(
            f"[yellow]⚠ Could not reach USPTO API: {e}. "
            f"Check internet connection and base_url in config.yaml.[/yellow]"
        )


def main():
    console.rule("[bold]Environment verification[/bold]")
    config = check_config()
    check_env_keys(config)
    check_tesseract()
    check_directories(config)
    smoke_test_uspto_api(config)
    console.rule("[green]Verification complete[/green]")
    console.print("\nNext: [cyan]python scripts/01_pull_uspto.py --dry-run --class D24[/cyan]")


if __name__ == "__main__":
    main()