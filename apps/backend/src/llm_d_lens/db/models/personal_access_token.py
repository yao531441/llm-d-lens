"""Revocable personal API credentials, separate from browser sessions."""

from datetime import datetime

from sqlalchemy import ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from llm_d_lens.auth.records import PersonalAccessTokenRecord
from llm_d_lens.db.base import Base, DeclarativeDtoMixin, UTCDateTime


class PersonalAccessTokenRow(DeclarativeDtoMixin, Base):
    __tablename__ = "personal_access_tokens"
    dto_type = PersonalAccessTokenRecord
    column_map = {
        name: name
        for name in (
            "id",
            "user_id",
            "name",
            "token_prefix",
            "token_hash",
            "created_at",
            "expires_at",
            "last_used_at",
            "revoked_at",
        )
    }

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    token_prefix: Mapped[str] = mapped_column(String(20), nullable=False)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    last_used_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)

    __table_args__ = (
        Index("uq_personal_access_tokens_hash", "token_hash", unique=True),
        Index("ix_personal_access_tokens_user", "user_id", "created_at"),
    )
