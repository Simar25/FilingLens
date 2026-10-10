"""Project settings: folder locations, plus values from config/companies.yaml and .env."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DB_PATH = PROJECT_ROOT / "data" / "processed" / "filinglens.duckdb"


@dataclass(frozen=True)
class Settings:
    tickers: list[str]
    start_fiscal_year: int
    end_fiscal_year: int
    benchmark: str
    sec_user_agent: str


def load_settings(config_path: Path = PROJECT_ROOT / "config" / "companies.yaml") -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Settings(
        tickers=[t.upper() for t in cfg["tickers"]],
        start_fiscal_year=int(cfg["start_fiscal_year"]),
        end_fiscal_year=int(cfg["end_fiscal_year"]),
        benchmark=cfg["benchmark"].upper(),
        sec_user_agent=os.getenv("SEC_USER_AGENT", "").strip(),
    )
