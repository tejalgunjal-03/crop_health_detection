from typing import Literal, Optional
from pydantic import BaseModel, Field


class RegisterIn(BaseModel):
    email: str = Field(min_length=5, max_length=255, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    name: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=6, max_length=128)


class LoginIn(BaseModel):
    email: str
    password: str


class SessionIn(BaseModel):
    monitoring_mode: Literal["UAV", "DIRECT_LEAF"]
    field_name: Optional[str] = None


class FusionIn(BaseModel):
    session_id: int
