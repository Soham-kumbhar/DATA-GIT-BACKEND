from datetime import datetime, timedelta, timezone
import secrets
from urllib.parse import urlencode

import jwt
import requests

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)
from fastapi.responses import RedirectResponse

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import (
    ALGORITHM,
    create_access_token,
    get_current_user,
    get_jwt_secret,
    hash_password,
    verify_password,
)

from app.db.database import get_db
from app.db.user_models import User

from app.schemas.auth import (
    AuthResponse,
    LoginRequest,
    OAuthExchangeRequest,
    RegisterRequest,
    UserResponse,
)

from app.services.cli_credential_service import (
    ensure_cli_token,
)


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


# ============================================================
# EMAIL
# ============================================================

def normalize_email(
    email: str,
) -> str:
    return email.strip().lower()


# ============================================================
# CLI ACCESS
# ============================================================

def _provision_cli_access(
    user: User,
) -> None:
    """
    Provision CLI authentication for the signed-in user.

    Browser authentication must continue to work even if
    operating-system credential storage is temporarily
    unavailable.
    """

    try:
        ensure_cli_token(
            user.id
        )
    except Exception:
        pass


# ============================================================
# COMMON AUTH RESPONSE
# ============================================================

def _build_auth_response(
    user: User,
) -> AuthResponse:
    access_token = create_access_token(
        user.id
    )

    _provision_cli_access(
        user
    )

    return AuthResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(
            id=user.id,
            name=user.name,
            email=user.email,
            created_at=user.created_at,
            updated_at=user.updated_at,
        ),
    )


# ============================================================
# NORMAL REGISTER
# ============================================================

@router.post(
    "/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(
    data: RegisterRequest,
    db: Session = Depends(get_db),
):
    name = data.name.strip()

    email = normalize_email(
        data.email
    )

    if not name:
        raise HTTPException(
            status_code=400,
            detail="Name cannot be empty.",
        )

    if "@" not in email:
        raise HTTPException(
            status_code=400,
            detail="Invalid email address.",
        )

    existing_user = (
        db.query(User)
        .filter(
            User.email == email
        )
        .first()
    )

    if existing_user:
        raise HTTPException(
            status_code=409,
            detail=(
                "An account with this email "
                "already exists."
            ),
        )

    user = User(
        name=name,
        email=email,
        password_hash=hash_password(
            data.password
        ),
    )

    db.add(user)

    try:
        db.commit()
        db.refresh(user)

    except Exception:
        db.rollback()
        raise

    return _build_auth_response(
        user
    )


# ============================================================
# NORMAL LOGIN
# ============================================================

@router.post(
    "/login",
    response_model=AuthResponse,
)
def login(
    data: LoginRequest,
    db: Session = Depends(get_db),
):
    email = normalize_email(
        data.email
    )

    user = (
        db.query(User)
        .filter(
            User.email == email
        )
        .first()
    )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    if not verify_password(
        data.password,
        user.password_hash,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    return _build_auth_response(
        user
    )


# ============================================================
# CURRENT USER
# ============================================================

@router.get(
    "/me",
    response_model=UserResponse,
)
def get_me(
    current_user: User = Depends(
        get_current_user
    ),
):
    _provision_cli_access(
        current_user
    )

    return UserResponse(
        id=current_user.id,
        name=current_user.name,
        email=current_user.email,
        created_at=current_user.created_at,
        updated_at=current_user.updated_at,
    )


# ============================================================
# OAUTH STATE
# ============================================================

OAUTH_STATE_EXPIRE_MINUTES = 10
OAUTH_TICKET_EXPIRE_MINUTES = 2

_used_oauth_tickets: set[str] = set()


def _create_oauth_state(
    provider: str,
) -> str:
    payload = {
        "typ": "oauth_state",
        "provider": provider,
        "nonce": secrets.token_urlsafe(32),
        "exp": (
            datetime.now(timezone.utc)
            + timedelta(
                minutes=OAUTH_STATE_EXPIRE_MINUTES
            )
        ),
    }

    return jwt.encode(
        payload,
        get_jwt_secret(),
        algorithm=ALGORITHM,
    )


def _validate_oauth_state(
    state: str,
    provider: str,
) -> None:
    try:
        payload = jwt.decode(
            state,
            get_jwt_secret(),
            algorithms=[ALGORITHM],
        )

        if payload.get("typ") != "oauth_state":
            raise ValueError(
                "Invalid OAuth state type."
            )

        if payload.get("provider") != provider:
            raise ValueError(
                "OAuth provider mismatch."
            )

    except (
        jwt.InvalidTokenError,
        ValueError,
        TypeError,
    ) as error:
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired OAuth state.",
        ) from error


# ============================================================
# OAUTH TICKET
# ============================================================

def _create_oauth_ticket(
    user_id: int,
) -> str:
    ticket_id = secrets.token_urlsafe(
        32
    )

    payload = {
        "typ": "oauth_ticket",
        "sub": str(user_id),
        "jti": ticket_id,
        "exp": (
            datetime.now(timezone.utc)
            + timedelta(
                minutes=OAUTH_TICKET_EXPIRE_MINUTES
            )
        ),
    }

    return jwt.encode(
        payload,
        get_jwt_secret(),
        algorithm=ALGORITHM,
    )


def _consume_oauth_ticket(
    ticket: str,
) -> int:
    try:
        payload = jwt.decode(
            ticket,
            get_jwt_secret(),
            algorithms=[ALGORITHM],
        )

        if payload.get("typ") != "oauth_ticket":
            raise ValueError(
                "Invalid OAuth ticket type."
            )

        ticket_id = payload.get(
            "jti"
        )

        if not ticket_id:
            raise ValueError(
                "OAuth ticket id is missing."
            )

        if ticket_id in _used_oauth_tickets:
            raise ValueError(
                "OAuth ticket has already been used."
            )

        user_id = int(
            payload["sub"]
        )

        _used_oauth_tickets.add(
            ticket_id
        )

        # Keep the in-memory collection bounded.
        if len(_used_oauth_tickets) > 5000:
            _used_oauth_tickets.clear()

        return user_id

    except (
        jwt.InvalidTokenError,
        ValueError,
        TypeError,
        KeyError,
    ) as error:
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired OAuth ticket.",
        ) from error


# ============================================================
# FRONTEND REDIRECT
# ============================================================

def _frontend_oauth_redirect(
    ticket: str | None = None,
    error: str | None = None,
) -> str:
    params: dict[str, str] = {}

    if ticket:
        params["ticket"] = ticket

    if error:
        params["error"] = error

    base_url = (
        settings.frontend_url
        .rstrip("/")
    )

    if not params:
        return (
            f"{base_url}/auth/callback"
        )

    return (
        f"{base_url}/auth/callback?"
        f"{urlencode(params)}"
    )


# ============================================================
# GOOGLE CONFIG
# ============================================================

def _google_redirect_uri() -> str:
    if settings.google_redirect_uri:
        return settings.google_redirect_uri

    return (
        settings.backend_url.rstrip("/")
        + "/auth/oauth/google/callback"
    )


def _github_redirect_uri() -> str:
    if settings.github_redirect_uri:
        return settings.github_redirect_uri

    return (
        settings.backend_url.rstrip("/")
        + "/auth/oauth/github/callback"
    )


# ============================================================
# GOOGLE START
# ============================================================

@router.get(
    "/oauth/google/start",
)
def google_start():
    if not settings.google_client_id:
        raise HTTPException(
            status_code=500,
            detail=(
                "Google OAuth is not configured. "
                "Set GOOGLE_CLIENT_ID in .env."
            ),
        )

    redirect_uri = (
        _google_redirect_uri()
    )

    state = _create_oauth_state(
        "google"
    )

    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "offline",
        "prompt": "select_account",
    }

    authorization_url = (
        "https://accounts.google.com/o/oauth2/v2/auth?"
        + urlencode(params)
    )

    return RedirectResponse(
        url=authorization_url,
        status_code=302,
    )


# ============================================================
# GOOGLE CALLBACK
# ============================================================

@router.get(
    "/oauth/google/callback",
)
def google_callback(
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    db: Session = Depends(get_db),
):
    if state is None:
        return RedirectResponse(
            _frontend_oauth_redirect(
                error="Google OAuth state is missing."
            ),
            status_code=302,
        )

    try:
        _validate_oauth_state(
            state,
            "google",
        )
    except HTTPException as exc:
        return RedirectResponse(
            _frontend_oauth_redirect(
                error=str(
                    exc.detail
                )
            ),
            status_code=302,
        )

    if error:
        return RedirectResponse(
            _frontend_oauth_redirect(
                error=(
                    "Google sign-in was cancelled."
                )
            ),
            status_code=302,
        )

    if not code:
        return RedirectResponse(
            _frontend_oauth_redirect(
                error=(
                    "Google OAuth did not return "
                    "an authorization code."
                )
            ),
            status_code=302,
        )

    if not settings.google_client_id:
        return RedirectResponse(
            _frontend_oauth_redirect(
                error=(
                    "Google OAuth is not configured."
                )
            ),
            status_code=302,
        )

    if not settings.google_client_secret:
        return RedirectResponse(
            _frontend_oauth_redirect(
                error=(
                    "Google OAuth client secret "
                    "is not configured."
                )
            ),
            status_code=302,
        )

    try:
        token_response = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": settings.google_client_id,
                "client_secret": (
                    settings.google_client_secret
                ),
                "code": code,
                "grant_type": (
                    "authorization_code"
                ),
                "redirect_uri": (
                    _google_redirect_uri()
                ),
            },
            timeout=15,
        )

        token_response.raise_for_status()

        token_data = (
            token_response.json()
        )

        google_access_token = token_data.get(
            "access_token"
        )

        if not google_access_token:
            raise ValueError(
                "Google did not return an access token."
            )

        userinfo_response = requests.get(
            "https://openidconnect.googleapis.com/v1/userinfo",
            headers={
                "Authorization": (
                    "Bearer "
                    + google_access_token
                )
            },
            timeout=15,
        )

        userinfo_response.raise_for_status()

        profile = (
            userinfo_response.json()
        )

        email = normalize_email(
            profile.get(
                "email",
                "",
            )
        )

        email_verified = bool(
            profile.get(
                "email_verified",
                False,
            )
        )

        provider_id = str(
            profile.get(
                "sub",
                "",
            )
        )

        if not email or not email_verified:
            raise ValueError(
                "Google did not provide a verified email address."
            )

        if not provider_id:
            raise ValueError(
                "Google account identifier is missing."
            )

        name = (
            profile.get("name")
            or email.split("@")[0]
        ).strip()

        user = _find_or_create_oauth_user(
            db=db,
            provider="google",
            provider_id=provider_id,
            email=email,
            name=name,
        )

        ticket = _create_oauth_ticket(
            user.id
        )

        return RedirectResponse(
            _frontend_oauth_redirect(
                ticket=ticket
            ),
            status_code=302,
        )

    except (
        requests.RequestException,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        return RedirectResponse(
            _frontend_oauth_redirect(
                error=(
                    "Google sign-in could not be completed: "
                    + str(exc)
                )
            ),
            status_code=302,
        )


# ============================================================
# GITHUB START
# ============================================================

@router.get(
    "/oauth/github/start",
)
def github_start():
    if not settings.github_client_id:
        raise HTTPException(
            status_code=500,
            detail=(
                "GitHub OAuth is not configured. "
                "Set GITHUB_CLIENT_ID in .env."
            ),
        )

    redirect_uri = (
        _github_redirect_uri()
    )

    state = _create_oauth_state(
        "github"
    )

    params = {
        "client_id": settings.github_client_id,
        "redirect_uri": redirect_uri,
        "scope": "read:user user:email",
        "state": state,
    }

    authorization_url = (
        "https://github.com/login/oauth/authorize?"
        + urlencode(params)
    )

    return RedirectResponse(
        url=authorization_url,
        status_code=302,
    )


# ============================================================
# GITHUB CALLBACK
# ============================================================

@router.get(
    "/oauth/github/callback",
)
def github_callback(
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    db: Session = Depends(get_db),
):
    if state is None:
        return RedirectResponse(
            _frontend_oauth_redirect(
                error="GitHub OAuth state is missing."
            ),
            status_code=302,
        )

    try:
        _validate_oauth_state(
            state,
            "github",
        )
    except HTTPException as exc:
        return RedirectResponse(
            _frontend_oauth_redirect(
                error=str(
                    exc.detail
                )
            ),
            status_code=302,
        )

    if error:
        return RedirectResponse(
            _frontend_oauth_redirect(
                error=(
                    "GitHub sign-in was cancelled."
                )
            ),
            status_code=302,
        )

    if not code:
        return RedirectResponse(
            _frontend_oauth_redirect(
                error=(
                    "GitHub OAuth did not return "
                    "an authorization code."
                )
            ),
            status_code=302,
        )

    if not settings.github_client_id:
        return RedirectResponse(
            _frontend_oauth_redirect(
                error=(
                    "GitHub OAuth is not configured."
                )
            ),
            status_code=302,
        )

    if not settings.github_client_secret:
        return RedirectResponse(
            _frontend_oauth_redirect(
                error=(
                    "GitHub OAuth client secret "
                    "is not configured."
                )
            ),
            status_code=302,
        )

    try:
        token_response = requests.post(
            "https://github.com/login/oauth/access_token",
            data={
                "client_id": settings.github_client_id,
                "client_secret": (
                    settings.github_client_secret
                ),
                "code": code,
                "redirect_uri": (
                    _github_redirect_uri()
                ),
            },
            headers={
                "Accept": "application/json",
            },
            timeout=15,
        )

        token_response.raise_for_status()

        token_data = (
            token_response.json()
        )

        github_access_token = (
            token_data.get(
                "access_token"
            )
        )

        if not github_access_token:
            raise ValueError(
                "GitHub did not return an access token."
            )

        github_headers = {
            "Authorization": (
                "Bearer "
                + github_access_token
            ),
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "DATAGIT",
        }

        user_response = requests.get(
            "https://api.github.com/user",
            headers=github_headers,
            timeout=15,
        )

        user_response.raise_for_status()

        profile = (
            user_response.json()
        )

        provider_id = str(
            profile.get(
                "id",
                "",
            )
        )

        if not provider_id:
            raise ValueError(
                "GitHub account identifier is missing."
            )

        email = normalize_email(
            profile.get(
                "email",
                "",
            )
            or ""
        )

        # GitHub may keep the user's email private.
        # Read verified emails when necessary.
        if not email:
            emails_response = requests.get(
                "https://api.github.com/user/emails",
                headers=github_headers,
                timeout=15,
            )

            emails_response.raise_for_status()

            email_entries = (
                emails_response.json()
            )

            verified_entries = [
                entry
                for entry in email_entries
                if entry.get("verified") is True
                and entry.get("email")
            ]

            primary_entries = [
                entry
                for entry in verified_entries
                if entry.get("primary") is True
            ]

            selected_entry = (
                primary_entries[0]
                if primary_entries
                else (
                    verified_entries[0]
                    if verified_entries
                    else None
                )
            )

            if selected_entry:
                email = normalize_email(
                    selected_entry["email"]
                )

        if not email:
            raise ValueError(
                "GitHub did not provide a verified email address."
            )

        name = (
            profile.get("name")
            or profile.get("login")
            or email.split("@")[0]
        ).strip()

        user = _find_or_create_oauth_user(
            db=db,
            provider="github",
            provider_id=provider_id,
            email=email,
            name=name,
        )

        ticket = _create_oauth_ticket(
            user.id
        )

        return RedirectResponse(
            _frontend_oauth_redirect(
                ticket=ticket
            ),
            status_code=302,
        )

    except (
        requests.RequestException,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        return RedirectResponse(
            _frontend_oauth_redirect(
                error=(
                    "GitHub sign-in could not be completed: "
                    + str(exc)
                )
            ),
            status_code=302,
        )


# ============================================================
# FIND / CREATE OAUTH USER
# ============================================================

def _find_or_create_oauth_user(
    db: Session,
    provider: str,
    provider_id: str,
    email: str,
    name: str,
) -> User:
    if provider == "google":
        user = (
            db.query(User)
            .filter(
                User.google_id == provider_id
            )
            .first()
        )

    elif provider == "github":
        user = (
            db.query(User)
            .filter(
                User.github_id == provider_id
            )
            .first()
        )

    else:
        raise ValueError(
            "Unsupported OAuth provider."
        )

    if user:
        if provider == "google":
            if user.google_id != provider_id:
                user.google_id = provider_id

        else:
            if user.github_id != provider_id:
                user.github_id = provider_id

        db.commit()
        db.refresh(user)

        return user

    # A verified OAuth email can connect to an
    # existing DATAGIT account.
    user = (
        db.query(User)
        .filter(
            User.email == email
        )
        .first()
    )

    if user:
        if provider == "google":
            user.google_id = provider_id
        else:
            user.github_id = provider_id

        db.commit()
        db.refresh(user)

        return user

    safe_name = (
        name.strip()
        or email.split("@")[0]
    )[:100]

    # OAuth-only accounts still need a non-null
    # password_hash because the existing schema requires it.
    random_password = secrets.token_urlsafe(
        48
    )

    user = User(
        name=safe_name,
        email=email,
        password_hash=hash_password(
            random_password
        ),
    )

    if provider == "google":
        user.google_id = provider_id
    else:
        user.github_id = provider_id

    db.add(user)

    try:
        db.commit()
        db.refresh(user)

    except Exception:
        db.rollback()
        raise

    return user


# ============================================================
# EXCHANGE OAUTH TICKET FOR NORMAL DATAGIT JWT
# ============================================================

@router.post(
    "/oauth/exchange",
    response_model=AuthResponse,
)
def exchange_oauth_ticket(
    data: OAuthExchangeRequest,
    db: Session = Depends(get_db),
):
    user_id = _consume_oauth_ticket(
        data.ticket
    )

    user = (
        db.query(User)
        .filter(
            User.id == user_id
        )
        .first()
    )

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="OAuth account no longer exists.",
        )

    return _build_auth_response(
        user
    )
