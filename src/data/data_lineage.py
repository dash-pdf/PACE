"""PACE: Provenance & Data Lineage Ledger.

Maintains an immutable, traceable record of provenance for all canonical records,
documenting source origin, raw source identifiers, timestamps, transformations,
and reconciliation outcomes.

Ensures:
1. Idempotent ingestion (re-running never duplicates canonical records).
2. Complete audit trail from raw API payload to simulation-ready feature.
3. Explicit traceability of winning source in cross-source reconciliations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.schemas import DATA_LINEAGE_SCHEMA
from src.data.storage import DEFAULT_BASE_DIR, load_parquet, save_parquet
from src.data.unified_contracts import DataLineageRecord

logger = logging.getLogger(__name__)


class DataLineageTracker:
    """Manages recording, querying, and persisting provenance audit records."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = base_dir or DEFAULT_BASE_DIR
        self._ledger: Dict[str, DataLineageRecord] = {}
        self._lineage_file = self.base_dir / "data" / "historical" / "data_lineage.parquet"
        self._load_existing_ledger()

    def _load_existing_ledger(self) -> None:
        """Load persisted lineage records from disk if present."""
        if self._lineage_file.exists():
            try:
                table = pq.read_table(self._lineage_file)
                df = table.to_pandas()
                for _, row in df.iterrows():
                    rec = DataLineageRecord(
                        canonical_record_id=str(row.get("canonical_record_id", "")),
                        domain=str(row.get("domain", "")),
                        canonical_event_id=str(row.get("canonical_event_id", "")),
                        source_api=str(row.get("source_api", "")),
                        source_record_id=str(row.get("source_record_id", "")),
                        ingestion_timestamp=str(row.get("ingestion_timestamp", "")),
                        transformation_pipeline=str(row.get("transformation_pipeline", "")),
                        reconciliation_outcome=str(row.get("reconciliation_outcome", "")),
                        selected_source=str(row.get("selected_source", "")),
                        selection_reason=str(row.get("selection_reason", "")),
                        pipeline_version=str(row.get("pipeline_version", "PACE-1.0-Stage5.6")),
                    )
                    self._ledger[rec.canonical_record_id] = rec
                logger.debug(f"Loaded {len(self._ledger)} lineage records from disk.")
            except Exception as e:
                logger.warning(f"Could not load lineage ledger: {e}")

    def record_lineage(
        self,
        canonical_record_id: str,
        domain: str,
        canonical_event_id: str,
        source_api: str,
        source_record_id: str = "",
        transformation_pipeline: str = "canonical_normalization",
        reconciliation_outcome: str = "STANDALONE",
        selected_source: Optional[str] = None,
        selection_reason: str = "Authoritative ingestion",
        ingestion_timestamp: Optional[str] = None,
    ) -> DataLineageRecord:
        """Record or update provenance lineage for a canonical record."""
        now = ingestion_timestamp or datetime.now(timezone.utc).isoformat()
        winning_source = selected_source or source_api

        record = DataLineageRecord(
            canonical_record_id=canonical_record_id,
            domain=domain,
            canonical_event_id=canonical_event_id,
            source_api=source_api,
            source_record_id=str(source_record_id),
            ingestion_timestamp=now,
            transformation_pipeline=transformation_pipeline,
            reconciliation_outcome=reconciliation_outcome,
            selected_source=winning_source,
            selection_reason=selection_reason,
        )
        self._ledger[canonical_record_id] = record
        return record

    def get_lineage(self, canonical_record_id: str) -> Optional[DataLineageRecord]:
        """Retrieve lineage record for a canonical entity or result."""
        return self._ledger.get(canonical_record_id)

    def list_lineage(
        self,
        domain: Optional[str] = None,
        source_api: Optional[str] = None,
        limit: int = 100,
    ) -> List[DataLineageRecord]:
        """List lineage records filtered by domain or source."""
        records = list(self._ledger.values())
        if domain:
            records = [r for r in records if r.domain == domain]
        if source_api:
            records = [r for r in records if source_api.lower() in r.source_api.lower()]
        return records[:limit]

    def save_ledger(self) -> Path:
        """Persist lineage ledger to Parquet idempotently."""
        if not self._ledger:
            logger.debug("Lineage ledger is empty; skipping write.")
            return self._lineage_file

        records_data = [r.to_dict() for r in self._ledger.values()]
        df = pd.DataFrame(records_data, columns=DATA_LINEAGE_SCHEMA.names)
        self._lineage_file.parent.mkdir(parents=True, exist_ok=True)
        save_parquet(df, self._lineage_file, schema=DATA_LINEAGE_SCHEMA)
        logger.info(f"Persisted {len(df)} lineage records to {self._lineage_file}")
        return self._lineage_file

    def get_provenance_summary(self) -> Dict[str, Any]:
        """Return aggregate statistics on source contributions and reconciliation outcomes."""
        source_counts: Dict[str, int] = {}
        domain_counts: Dict[str, int] = {}
        outcome_counts: Dict[str, int] = {}

        for r in self._ledger.values():
            source_counts[r.source_api] = source_counts.get(r.source_api, 0) + 1
            domain_counts[r.domain] = domain_counts.get(r.domain, 0) + 1
            outcome_counts[r.reconciliation_outcome] = outcome_counts.get(r.reconciliation_outcome, 0) + 1

        return {
            "total_records": len(self._ledger),
            "sources": source_counts,
            "domains": domain_counts,
            "outcomes": outcome_counts,
        }
