"""Demo authentication and admin for LinguaGap desktop interface."""

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from starlette.requests import Request

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths for persistent storage
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
ACCOUNTS_FILE = DATA_DIR / "accounts.json"
LOGOS_DIR = DATA_DIR / "logos"

# Bundled default accounts (shipped with the image)
_BUNDLED_ACCOUNTS = Path(__file__).parent / "demo_accounts.json"

# ---------------------------------------------------------------------------
# Admin credentials
# ---------------------------------------------------------------------------

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@linguagap.local")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")


def verify_admin(email: str, password: str) -> bool:
    if not ADMIN_PASSWORD:
        return False
    return email == ADMIN_EMAIL and password == ADMIN_PASSWORD


def is_admin(request: Request) -> bool:
    return bool(request.session.get("is_admin"))


# ---------------------------------------------------------------------------
# Demo account model
# ---------------------------------------------------------------------------


@dataclass
class DemoAccount:
    email: str
    password: str
    display_name: str
    logo_url: str


# ---------------------------------------------------------------------------
# Account storage (persistent)
# ---------------------------------------------------------------------------

_accounts: list[DemoAccount] | None = None


def _load_accounts() -> list[DemoAccount]:
    """Load accounts: DEMO_ACCOUNTS env > /data/accounts.json > bundled default."""
    env_accounts = os.getenv("DEMO_ACCOUNTS")
    if env_accounts:
        raw = json.loads(env_accounts)
    elif ACCOUNTS_FILE.exists():
        raw = json.loads(ACCOUNTS_FILE.read_text())
    elif _BUNDLED_ACCOUNTS.exists():
        raw = json.loads(_BUNDLED_ACCOUNTS.read_text())
    else:
        logger.warning("No demo accounts configured")
        return []
    return [DemoAccount(**a) for a in raw]


def get_accounts() -> list[DemoAccount]:
    global _accounts  # noqa: PLW0603
    if _accounts is None:
        _accounts = _load_accounts()
    return _accounts


def save_accounts(accounts: list[DemoAccount]) -> None:
    """Persist accounts to disk and update cache."""
    global _accounts  # noqa: PLW0603
    ACCOUNTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    ACCOUNTS_FILE.write_text(json.dumps([asdict(a) for a in accounts], indent=2) + "\n")
    _accounts = accounts


def reload_accounts() -> list[DemoAccount]:
    """Clear cache and reload from disk."""
    global _accounts  # noqa: PLW0603
    _accounts = None
    return get_accounts()


def verify_credentials(email: str, password: str) -> tuple[DemoAccount | None, bool]:
    """Verify login credentials. Returns (account, is_admin)."""
    if verify_admin(email, password):
        return DemoAccount(
            email=ADMIN_EMAIL, password="", display_name="Admin", logo_url="/static/synia-logo.png"
        ), True
    for account in get_accounts():
        if account.email == email and account.password == password:
            return account, False
    return None, False


def get_current_user(request: Request) -> dict | None:
    email = request.session.get("email")
    if email:
        return {
            "email": email,
            "display_name": request.session.get("display_name", ""),
            "logo_url": request.session.get("logo_url", ""),
            "is_admin": bool(request.session.get("is_admin")),
        }
    return None
