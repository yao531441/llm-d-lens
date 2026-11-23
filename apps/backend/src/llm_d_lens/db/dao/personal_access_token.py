"""Persistence boundary for personal access tokens; never stores plaintext."""

from datetime import datetime

from sqlalchemy import select, update

from llm_d_lens.auth.records import PersonalAccessTokenRecord, utcnow
from llm_d_lens.db.dao.base import BaseDao
from llm_d_lens.db.models.personal_access_token import PersonalAccessTokenRow


class PersonalAccessTokenDao(BaseDao):
    def create(self, record: PersonalAccessTokenRecord) -> PersonalAccessTokenRecord:
        with self._transaction() as session:
            row = PersonalAccessTokenRow.from_dto(record)
            session.add(row)
            session.flush()
            return row.to_dto()

    def get_by_hash(self, token_hash: str) -> PersonalAccessTokenRecord | None:
        with self._read_only() as session:
            row = session.scalar(select(PersonalAccessTokenRow).where(PersonalAccessTokenRow.token_hash == token_hash))
            return row.to_dto() if row else None

    def list_for_user(self, user_id: str) -> list[PersonalAccessTokenRecord]:
        with self._read_only() as session:
            rows = session.scalars(
                select(PersonalAccessTokenRow)
                .where(PersonalAccessTokenRow.user_id == user_id)
                .order_by(PersonalAccessTokenRow.created_at.desc(), PersonalAccessTokenRow.id)
            )
            return [row.to_dto() for row in rows]

    def revoke(self, token_id: str, user_id: str) -> bool:
        with self._transaction() as session:
            row = session.get(PersonalAccessTokenRow, token_id)
            if row is None or row.user_id != user_id:
                return False
            if row.revoked_at is None:
                row.revoked_at = utcnow()
            return True

    def revoke_for_user(self, user_id: str) -> None:
        with self._transaction() as session:
            session.execute(
                update(PersonalAccessTokenRow)
                .where(PersonalAccessTokenRow.user_id == user_id, PersonalAccessTokenRow.revoked_at.is_(None))
                .values(revoked_at=utcnow())
            )

    def touch(self, token_id: str, when: datetime) -> None:
        with self._transaction() as session:
            session.execute(
                update(PersonalAccessTokenRow)
                .where(PersonalAccessTokenRow.id == token_id, PersonalAccessTokenRow.revoked_at.is_(None))
                .values(last_used_at=when)
            )
