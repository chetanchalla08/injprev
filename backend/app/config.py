"""Runtime configuration, read once from the environment (and backend/.env if present)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # python-dotenv ships with uvicorn[standard]; the app works without it.
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:  # pragma: no cover
    pass

BACKEND_DIR = Path(__file__).resolve().parent.parent


def _path(name: str, default: str) -> Path:
    p = Path(os.environ.get(name, default))
    return p if p.is_absolute() else (BACKEND_DIR / p).resolve()


@dataclass
class Settings:
    data_dir: Path
    model_dir: Path
    pose_model: str
    max_upload_mb: int
    max_duration_s: float
    min_duration_s: float
    analysis_max_fps: float
    retention_days: int
    max_attempts: int
    cookie_secure: bool
    cors_origins: list[str]
    enable_demo: bool = False
    data_key: str = ""  # base64 32-byte key for encryption at rest; empty = stored unencrypted
    require_encryption: bool = False  # refuse to start without DATA_KEY
    reviewer_code: str = ""  # shared code a reviewer must enter to sign off evidence; empty = sign-off disabled
    share_ttl_days: int = 7
    share_max_ttl_days: int = 30
    llm_api_key: str = ""  # optional; the app works fully without it
    llm_model: str = "claude-opus-5-5"
    pose_mode: str = "full_frame"
    enable_ball: bool = True
    enable_court: bool = True
    enable_racket: bool = False

    @property
    def model_path(self) -> Path:
        return self.model_dir / f"pose_landmarker_{self.pose_model}.task"

    @property
    def object_model_path(self) -> Path:
        return self.model_dir / "yolox_s.onnx"

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"

    def options(self):
        from .pipeline.options import Options

        return Options(pose_mode=self.pose_mode, ball=self.enable_ball, court=self.enable_court, racket=self.enable_racket and self.object_model_path.exists())

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.sqlite3"

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"


def load_settings() -> Settings:
    return Settings(
        data_dir=_path("DATA_DIR", "./data"),
        model_dir=_path("MODEL_DIR", "./models"),
        pose_model=os.environ.get("POSE_MODEL", "heavy"),
        max_upload_mb=int(os.environ.get("MAX_UPLOAD_MB", "300")),
        max_duration_s=float(os.environ.get("MAX_DURATION_S", "180")),
        min_duration_s=float(os.environ.get("MIN_DURATION_S", "1")),
        analysis_max_fps=float(os.environ.get("ANALYSIS_MAX_FPS", "60")),
        retention_days=int(os.environ.get("RETENTION_DAYS", "30")),
        max_attempts=int(os.environ.get("MAX_ATTEMPTS", "2")),
        cookie_secure=os.environ.get("COOKIE_SECURE", "0") == "1",
        enable_demo=os.environ.get("ENABLE_DEMO", "0") == "1",
        data_key=os.environ.get("DATA_KEY", ""),
        require_encryption=os.environ.get("REQUIRE_ENCRYPTION", "0") == "1",
        reviewer_code=os.environ.get("REVIEWER_CODE", ""),
        share_ttl_days=int(os.environ.get("SHARE_TTL_DAYS", "7")),
        share_max_ttl_days=int(os.environ.get("SHARE_MAX_TTL_DAYS", "30")),
        llm_api_key=os.environ.get("ANTHROPIC_API_KEY", "") if os.environ.get("ENABLE_LLM_SUMMARY", "0") == "1" else "",
        llm_model=os.environ.get("LLM_MODEL", "claude-opus-5-5"),
        pose_mode=os.environ.get("POSE_MODE", "full_frame"),
        enable_ball=os.environ.get("ENABLE_BALL", "1") == "1",
        enable_court=os.environ.get("ENABLE_COURT", "1") == "1",
        enable_racket=os.environ.get("ENABLE_RACKET", "0") == "1",
        cors_origins=[o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()],
    )
