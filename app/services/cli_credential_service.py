import logging

import keyring

from app.core.security import (
    create_cli_access_token,
    decode_access_token,
)


LOGGER = logging.getLogger(__name__)

KEYRING_SERVICE = "DATAGIT"

ACTIVE_USER_KEY = "active-user-id"

CLI_TOKEN_PREFIX = "cli-token:"


def _user_token_key(
    user_id: int,
) -> str:
    return (
        f"{CLI_TOKEN_PREFIX}{user_id}"
    )


def _read_password(
    key: str,
) -> str | None:
    try:
        return keyring.get_password(
            KEYRING_SERVICE,
            key,
        )
    except Exception as error:
        LOGGER.warning(
            "DATAGIT secure credential storage "
            "could not be read: %s",
            error,
        )
        return None


def _write_password(
    key: str,
    value: str,
) -> bool:
    try:
        keyring.set_password(
            KEYRING_SERVICE,
            key,
            value,
        )
        return True
    except Exception as error:
        LOGGER.warning(
            "DATAGIT secure credential storage "
            "could not be written: %s",
            error,
        )
        return False


def _delete_password(
    key: str,
) -> None:
    try:
        keyring.delete_password(
            KEYRING_SERVICE,
            key,
        )
    except Exception:
        pass


def ensure_cli_token(
    user_id: int,
) -> str | None:
    """
    Ensure the current operating-system user has a
    DATAGIT CLI credential for the supplied DATAGIT user.

    The credential is stored in the OS credential store,
    never in the project directory and never printed.
    """

    token_key = _user_token_key(user_id)

    existing_token = _read_password(
        token_key
    )

    if existing_token:
        try:
            token_user_id = decode_access_token(
                existing_token
            )

            if token_user_id == user_id:
                _write_password(
                    ACTIVE_USER_KEY,
                    str(user_id),
                )

                return existing_token

        except ValueError:
            # Existing CLI token is expired/invalid.
            # Generate a fresh one below.
            pass

    new_token = create_cli_access_token(
        user_id
    )

    stored = _write_password(
        token_key,
        new_token,
    )

    if not stored:
        return None

    active_stored = _write_password(
        ACTIVE_USER_KEY,
        str(user_id),
    )

    if not active_stored:
        return None

    return new_token


def get_active_cli_token() -> str | None:
    """
    Return the CLI credential belonging to the
    currently active DATAGIT account.
    """

    active_user_id = _read_password(
        ACTIVE_USER_KEY
    )

    if not active_user_id:
        return None

    try:
        user_id = int(active_user_id)
    except ValueError:
        return None

    token = _read_password(
        _user_token_key(user_id)
    )

    if not token:
        return None

    try:
        token_user_id = decode_access_token(
            token
        )
    except ValueError:
        _delete_password(
            _user_token_key(user_id)
        )
        return None

    if token_user_id != user_id:
        _delete_password(
            _user_token_key(user_id)
        )
        return None

    return token


def clear_active_cli_token() -> None:
    """
    Remove the currently active CLI credential.
    """

    active_user_id = _read_password(
        ACTIVE_USER_KEY
    )

    if not active_user_id:
        return

    try:
        user_id = int(active_user_id)
    except ValueError:
        _delete_password(
            ACTIVE_USER_KEY
        )
        return

    _delete_password(
        _user_token_key(user_id)
    )

    _delete_password(
        ACTIVE_USER_KEY
    )