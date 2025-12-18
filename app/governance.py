"""
Governance Engine Module.

Implements the "Policy as Code" mechanism for the Data Mesh architecture.
Evaluates data compliance against the formal Data Contract defined in YAML.
"""

import logging
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ComplianceReport:
    """Report on data contract compliance."""
    score: float
    violations: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "PENDING"  # PASSED, WARNING, BLOCKED

    def to_dict(self):
        return {
            "score": self.score,
            "violations": self.violations,
            "status": self.status
        }

class GovernanceEngine:
    """
    Enforces the Data Contract policies on datasets.
    Acting as the 'Logic Unit' of the APU Node.
    """

    def __init__(self, contract_path: str = "config/data_contract.yaml"):
        self.contract_path = Path(contract_path)
        self.contract = self.load_contract(self.contract_path)

    def load_contract(self, path: Path) -> Dict:
        """Loads the Data Contract YAML file."""
        if not path.exists():
            logger.error(f"❌ Data Contract not found at {path}")
            return {}

        try:
            with open(path, 'r', encoding='utf-8') as f:
                contract = yaml.safe_load(f)
            logger.info(f"📜 Data Contract loaded: {contract.get('domain', 'Unknown Domain')}")
            return contract
        except Exception as e:
            logger.error(f"❌ Error loading Data Contract: {e}")
            return {}

    def enforce_policy(self, df: pd.DataFrame, dataset_name: str) -> ComplianceReport:
        """
        Validates the DataFrame against the contract policies.

        Args:
            df: The DataFrame to validate.
            dataset_name: Name of the dataset for context.

        Returns:
            ComplianceReport object.
        """
        if self.contract is None or not self.contract:
             logger.warning("⚠️ No valid contract loaded. Skipping governance check.")
             return ComplianceReport(score=100.0, status="SKIPPED")

        violations = []
        total_checks = 0
        passed_checks = 0

        # --- Schema Policy ---
        schema_policy = self.contract.get("schema_policy", {})
        required_cols = schema_policy.get("required_columns", [])

        for col in required_cols:
            total_checks += 1
            if col not in df.columns:
                violations.append({
                    "policy": "schema",
                    "check": "required_columns",
                    "detail": f"Missing column: {col}",
                    "severity": "CRITICAL"
                })
            else:
                passed_checks += 1

        # --- Quality Policy ---
        quality_policy = self.contract.get("quality_policy", {})
        max_null_pct = quality_policy.get("max_null_percentage", 5.0)
        allow_negative = quality_policy.get("allow_negative_costs", False)
        allow_zero_qty = quality_policy.get("allow_zero_quantities", False)

        # Check Nulls
        total_checks += 1
        null_pct = df.isnull().mean().max() * 100 # Worst column
        if null_pct > max_null_pct:
            violations.append({
                "policy": "quality",
                "check": "max_null_percentage",
                "detail": f"Max null percentage {null_pct:.2f}% exceeds limit {max_null_pct}%",
                "severity": "WARNING"
            })
        else:
            passed_checks += 1

        # Check Negative Costs (if applicable columns exist)
        cost_cols = [c for c in df.columns if "VALOR" in c or "COSTO" in c or "PRECIO" in c]
        if cost_cols and not allow_negative:
             total_checks += 1
             has_negatives = False
             for col in cost_cols:
                 if pd.api.types.is_numeric_dtype(df[col]):
                     if (df[col] < 0).any():
                         has_negatives = True
                         break

             if has_negatives:
                 violations.append({
                     "policy": "quality",
                     "check": "allow_negative_costs",
                     "detail": "Negative values found in cost columns",
                     "severity": "CRITICAL"
                 })
             else:
                 passed_checks += 1

        # Check Zero Quantities
        qty_cols = [c for c in df.columns if "CANTIDAD" in c]
        if qty_cols and not allow_zero_qty:
            total_checks += 1
            has_zeros = False
            for col in qty_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    if (df[col] == 0).any():
                        has_zeros = True
                        break

            if has_zeros:
                violations.append({
                    "policy": "quality",
                    "check": "allow_zero_quantities",
                    "detail": "Zero quantities found",
                    "severity": "WARNING"
                })
            else:
                passed_checks += 1

        # --- Semantic Policy ---
        semantic_policy = self.contract.get("semantic_policy", {})
        min_desc_len = semantic_policy.get("description_min_length", 0)
        forbidden = semantic_policy.get("forbidden_terms", [])

        desc_cols = [c for c in df.columns if "DESCRIPCION" in c]

        # Check Description Length
        if desc_cols and min_desc_len > 0:
            total_checks += 1
            failed_rows = 0
            for col in desc_cols:
                # Convert to str, measure len
                lens = df[col].astype(str).str.len()
                failed_rows += (lens < min_desc_len).sum()

            if failed_rows > 0:
                 violations.append({
                    "policy": "semantic",
                    "check": "description_min_length",
                    "detail": f"{failed_rows} rows have descriptions shorter than {min_desc_len} chars",
                    "severity": "WARNING"
                })
            else:
                passed_checks += 1

        # Check Forbidden Terms
        if desc_cols and forbidden:
            total_checks += 1
            found_terms = []
            for col in desc_cols:
                for term in forbidden:
                    # Case insensitive search
                    if df[col].astype(str).str.contains(term, case=False, regex=False).any():
                        found_terms.append(term)

            if found_terms:
                 violations.append({
                    "policy": "semantic",
                    "check": "forbidden_terms",
                    "detail": f"Found forbidden terms: {list(set(found_terms))}",
                    "severity": "CRITICAL"
                })
            else:
                passed_checks += 1


        # Calculate Score
        score = (passed_checks / total_checks * 100) if total_checks > 0 else 100.0

        # Determine Status
        status = "PASSED"
        for v in violations:
            if v['severity'] == "CRITICAL":
                status = "BLOCKED" # Logic Unit decision: Hard fail criteria
            elif v['severity'] == "WARNING" and status != "BLOCKED":
                status = "WARNING"

        return ComplianceReport(
            score=score,
            violations=violations,
            status=status
        )
