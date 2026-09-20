from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from sqlalchemy.orm import Session

from app.core.security import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)

from app.db.database import get_db
from app.db.user_models import User

from app.schemas.auth import (
    AuthResponse,
    LoginRequest,
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


def normalize_email(
    email: str,
) -> str:
    return email.strip().lower()


def _provision_cli_access(
    user: User,
) -> None:
    """
    Provision CLI authentication for the signed-in user.

    The credential is stored by the CLI credential service.
    Failure to provision CLI access must not break browser
    authentication.
    """

    try:
        ensure_cli_token(
            user.id
        )
    except Exception:
        # Browser login must remain functional even if
        # OS credential storage is temporarily unavailable.
        pass


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

    access_token = create_access_token(
        user.id
    )

    # Provision CLI authentication silently.
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

    access_token = create_access_token(
        user.id
    )

    # Provision CLI authentication silently.
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


@router.get(
    "/me",
    response_model=UserResponse,
)
def get_me(
    current_user: User = Depends(
        get_current_user
    ),
):
    # Make sure an existing signed-in user also gets
    # CLI access without manually copying a token.
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