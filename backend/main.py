import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from db import engine, Base
from models import models  # noqa: F401 — registers all ORM models
from routers import auth, notas, dashboard
from routers.auth import get_password_hash
from db import SessionLocal
from models.models import User, UserRole

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s")

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Leitor de Nota Fiscal",
    description="API para leitura e gestão de NF-e, NFC-e, NFS-e e comprovantes PIX",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(notas.router)
app.include_router(dashboard.router)


@app.on_event("startup")
async def seed_admin():
    """Create default admin if no users exist. Also runs any pending migrations."""
    # Migration: add confirmado column if it doesn't exist (SQLite ALTER TABLE)
    with engine.connect() as conn:
        try:
            conn.execute(text(
                "ALTER TABLE notas_fiscais ADD COLUMN confirmado INTEGER NOT NULL DEFAULT 0"
            ))
            # Records that existed before this migration were already reviewed — confirm them
            conn.execute(text("UPDATE notas_fiscais SET confirmado = 1"))
            conn.commit()
        except Exception:
            pass  # Column already exists

    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            db.add(User(
                email="admin@empresa.com",
                hashed_password=get_password_hash("admin123"),
                role=UserRole.admin,
            ))
            db.commit()
            logging.getLogger(__name__).info(
                "Admin criado: admin@empresa.com / admin123"
            )
    finally:
        db.close()


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}
