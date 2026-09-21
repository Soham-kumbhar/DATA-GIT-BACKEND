from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class RegisterRequest(BaseModel):
    name: str = Field(
        min_length=2,
        max_length=100,
    )

    email: str = Field(
        min_length=5,
        max_length=320,
    )

    password: str = Field(
        min_length=8,
        max_length=128,
    )


class LoginRequest(BaseModel):
    email: str = Field(
        min_length=5,
        max_length=320,
    )

    password: str = Field(
        min_length=1,
        max_length=128,
    )


class OAuthExchangeRequest(BaseModel):
    ticket: str = Field(
        min_length=1,
        max_length=4096,
    )


class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
    )


class AuthResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse
