from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timezone
from db import get_db
from models.models import NotaFiscal, DocumentType, User
from schemas.schemas import DashboardStats
from routers.auth import get_current_user

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/stats", response_model=DashboardStats)
async def get_stats(db: Session = Depends(get_db),
                    _: User = Depends(get_current_user)):
    now = datetime.now(timezone.utc)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    confirmed = NotaFiscal.confirmado == True  # noqa: E712

    total_mes = db.query(func.sum(NotaFiscal.valor_total)).filter(
        confirmed, NotaFiscal.data_importacao >= month_start).scalar() or 0.0

    total_pix = db.query(func.sum(NotaFiscal.valor_total)).filter(
        confirmed, NotaFiscal.tipo == DocumentType.pix,
        NotaFiscal.data_importacao >= month_start).scalar() or 0.0

    total_nf = db.query(func.sum(NotaFiscal.valor_total)).filter(
        confirmed, NotaFiscal.tipo.in_([DocumentType.nfe, DocumentType.nfce, DocumentType.nfse]),
        NotaFiscal.data_importacao >= month_start).scalar() or 0.0

    aguardando = db.query(func.count(NotaFiscal.id)).filter(
        confirmed, NotaFiscal.status_sefaz == "nao_validado").scalar() or 0

    total_docs = db.query(func.count(NotaFiscal.id)).filter(confirmed).scalar() or 0

    return DashboardStats(
        total_mes=total_mes, total_pix=total_pix, total_nf=total_nf,
        aguardando_revisao=aguardando, total_documentos=total_docs,
    )
