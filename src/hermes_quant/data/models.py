from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4


@dataclass(frozen=True)
class IngestionBatch:
    batch_id: UUID
    provider: str
    endpoint: str
    request_parameters: dict[str, str]
    retrieved_at: datetime
    schema_version: str
    row_count: int
    source_metadata: dict[str, str]

    @classmethod
    def create(
        cls,
        *,
        provider: str,
        endpoint: str,
        request_parameters: dict[str, str],
        retrieved_at: datetime,
        schema_version: str,
        row_count: int,
        source_metadata: dict[str, str],
    ) -> IngestionBatch:
        return cls(
            batch_id=uuid4(),
            provider=provider,
            endpoint=endpoint,
            request_parameters=request_parameters,
            retrieved_at=retrieved_at,
            schema_version=schema_version,
            row_count=row_count,
            source_metadata=source_metadata,
        )

    def write_json(self, path: Path) -> None:
        payload = asdict(self)
        payload["batch_id"] = str(self.batch_id)
        payload["retrieved_at"] = self.retrieved_at.isoformat()
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
