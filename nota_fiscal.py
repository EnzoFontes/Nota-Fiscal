#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Leitor de Nota Fiscal — Arquivo Único
  Stack: FastAPI + SQLAlchemy + Tesseract/OpenCV + React (CDN)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SETUP (instale as dependências):
    pip install fastapi "uvicorn[standard]" sqlalchemy
                passlib[bcrypt] "python-jose[cryptography]"
                python-multipart httpx pillow

  Para OCR de imagens (opcional mas recomendado):
    pip install pytesseract opencv-python-headless numpy
    Windows → Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
              Marque "Portuguese" no instalador
    Ubuntu  → sudo apt install tesseract-ocr tesseract-ocr-por

  RODAR LOCALMENTE:
    python nota_fiscal.py
    Acesse: http://localhost:8000
    Login:  admin@empresa.com / admin123

  DEPLOY (Render.com):
    1. Faça git push deste arquivo + requirements.txt
    2. Crie Web Service: Start Command → python nota_fiscal.py
    3. Crie PostgreSQL managed e conecte via DATABASE_URL env var
    4. Defina SECRET_KEY como variável de ambiente segura
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  requirements.txt sugerido:
    fastapi
    uvicorn[standard]
    sqlalchemy
    passlib[bcrypt]
    python-jose[cryptography]
    python-multipart
    httpx
    pillow
    pytesseract
    opencv-python-headless
    numpy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ─────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────
import os, io, re, csv, uuid, logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import xml.etree.ElementTree as ET

from fastapi import (
    FastAPI, Depends, HTTPException, UploadFile,
    File, status, Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    DateTime, ForeignKey, Text,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session

from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel

import httpx

# OCR — carregado dinamicamente para não quebrar se não instalado
try:
    import cv2
    import numpy as np
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
    # Caminho padrão do Tesseract no Windows
    if os.name == "nt":
        _tess = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(_tess):
            pytesseract.pytesseract.tesseract_cmd = _tess
except ImportError:
    OCR_AVAILABLE = False
    logging.warning(
        "OCR não disponível. Instale: pip install pytesseract opencv-python-headless numpy pillow"
    )

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nota_fiscal")

# ─────────────────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────────────────
# Banco local SQLite (para desenvolvimento); PostgreSQL via DATABASE_URL no Render
_RAW_DB_URL = os.getenv("DATABASE_URL", "sqlite:///./nota_fiscal.db")
# Render usa "postgres://", SQLAlchemy precisa de "postgresql://"
DATABASE_URL = _RAW_DB_URL.replace("postgres://", "postgresql://", 1)

SECRET_KEY = os.getenv("SECRET_KEY", "TROQUE_ISSO_EM_PRODUCAO_" + uuid.uuid4().hex)
ALGORITHM  = "HS256"
TOKEN_EXPIRE_HOURS = 8

# Confiança mínima do Tesseract para aceitar imagem (0–100)
OCR_CONFIDENCE_THRESHOLD = 55
# Timeout para consulta SEFAZ em segundos
SEFAZ_TIMEOUT = 5

# ─────────────────────────────────────────────────────
# BANCO DE DADOS
# ─────────────────────────────────────────────────────
_connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─────────────────────────────────────────────────────
# MODELOS (SQLAlchemy)
# ─────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"
    id              = Column(Integer, primary_key=True, index=True)
    email           = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role            = Column(String, default="viewer")   # "admin" | "viewer"
    created_at      = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    notas           = relationship("NotaFiscalDB", back_populates="importador")


class NotaFiscalDB(Base):
    __tablename__ = "notas_fiscais"
    id                    = Column(Integer, primary_key=True, index=True)
    tipo                  = Column(String, nullable=False)  # nfe | nfce | nfse
    numero                = Column(String)
    serie                 = Column(String)
    chave_acesso          = Column(String, unique=True, index=True)
    data_emissao          = Column(DateTime)
    data_importacao       = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    emitente_cnpj         = Column(String)
    emitente_razao_social = Column(String)
    destinatario_cpf_cnpj = Column(String)
    destinatario_nome     = Column(String)
    valor_total           = Column(Float)
    valor_desconto        = Column(Float, default=0.0)
    valor_frete           = Column(Float, default=0.0)
    status_sefaz          = Column(String, default="nao_validado")  # validado | nao_validado | falha
    confidence_score      = Column(Float)
    importado_por_id      = Column(Integer, ForeignKey("users.id"))
    importador            = relationship("User", back_populates="notas")
    itens                 = relationship("ItemNotaDB",    back_populates="nota", cascade="all, delete-orphan")
    tributos              = relationship("TributoNotaDB", back_populates="nota", cascade="all, delete-orphan")


class ItemNotaDB(Base):
    __tablename__ = "itens_nota"
    id             = Column(Integer, primary_key=True, index=True)
    nota_id        = Column(Integer, ForeignKey("notas_fiscais.id"))
    descricao      = Column(String)
    ncm            = Column(String)
    quantidade     = Column(Float)
    unidade        = Column(String)
    valor_unitario = Column(Float)
    valor_total    = Column(Float)
    cfop           = Column(String)
    categoria      = Column(String)
    nota           = relationship("NotaFiscalDB", back_populates="itens")


class TributoNotaDB(Base):
    __tablename__ = "tributos_nota"
    id           = Column(Integer, primary_key=True, index=True)
    nota_id      = Column(Integer, ForeignKey("notas_fiscais.id"))
    tipo         = Column(String)   # ICMS, PIS, COFINS, ISS …
    base_calculo = Column(Float)
    aliquota     = Column(Float)
    valor        = Column(Float)
    nota         = relationship("NotaFiscalDB", back_populates="tributos")


class ComprovantePixDB(Base):
    __tablename__ = "comprovantes_pix"
    id                  = Column(Integer, primary_key=True, index=True)
    valor               = Column(Float)
    data_hora           = Column(DateTime)
    data_importacao     = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    nome_pagador        = Column(String)
    cpf_cnpj_pagador    = Column(String)
    banco_pagador       = Column(String)
    nome_recebedor      = Column(String)
    cpf_cnpj_recebedor  = Column(String)
    banco_recebedor     = Column(String)
    chave_pix           = Column(String)
    tipo_chave          = Column(String)   # cpf | cnpj | telefone | email | aleatoria
    id_transacao        = Column(String, index=True)   # End-to-End ID (E + 32 chars)
    descricao           = Column(Text)
    confidence_score    = Column(Float)
    importado_por_id    = Column(Integer, ForeignKey("users.id"))
    importador          = relationship("User")


# ─────────────────────────────────────────────────────
# SCHEMAS (Pydantic)
# ─────────────────────────────────────────────────────
class Token(BaseModel):
    access_token: str
    token_type: str
    role: str


class UserCreate(BaseModel):
    email: str
    password: str
    role: str = "viewer"


class UserOut(BaseModel):
    id: int
    email: str
    role: str
    class Config:
        from_attributes = True


class ItemOut(BaseModel):
    descricao:      Optional[str]   = None
    ncm:            Optional[str]   = None
    quantidade:     Optional[float] = None
    unidade:        Optional[str]   = None
    valor_unitario: Optional[float] = None
    valor_total:    Optional[float] = None
    cfop:           Optional[str]   = None
    categoria:      Optional[str]   = None
    class Config:
        from_attributes = True


class TributoOut(BaseModel):
    tipo:         Optional[str]   = None
    base_calculo: Optional[float] = None
    aliquota:     Optional[float] = None
    valor:        Optional[float] = None
    class Config:
        from_attributes = True


class NotaFiscalOut(BaseModel):
    id:                    int
    tipo:                  str
    numero:                Optional[str]      = None
    serie:                 Optional[str]      = None
    chave_acesso:          Optional[str]      = None
    data_emissao:          Optional[datetime] = None
    data_importacao:       datetime
    emitente_cnpj:         Optional[str]      = None
    emitente_razao_social: Optional[str]      = None
    destinatario_cpf_cnpj: Optional[str]      = None
    destinatario_nome:     Optional[str]      = None
    valor_total:           Optional[float]    = None
    valor_desconto:        Optional[float]    = None
    valor_frete:           Optional[float]    = None
    status_sefaz:          str
    confidence_score:      Optional[float]    = None
    itens:                 List[ItemOut]      = []
    tributos:              List[TributoOut]   = []
    class Config:
        from_attributes = True


class NotaFiscalCreate(BaseModel):
    tipo:                  str
    numero:                Optional[str]      = None
    serie:                 Optional[str]      = None
    chave_acesso:          Optional[str]      = None
    data_emissao:          Optional[str]      = None   # ISO string do frontend
    emitente_cnpj:         Optional[str]      = None
    emitente_razao_social: Optional[str]      = None
    destinatario_cpf_cnpj: Optional[str]      = None
    destinatario_nome:     Optional[str]      = None
    valor_total:           Optional[float]    = None
    valor_desconto:        Optional[float]    = None
    valor_frete:           Optional[float]    = None
    confidence_score:      Optional[float]    = None
    itens:                 List[dict]         = []
    tributos:              List[dict]         = []


class DashboardStats(BaseModel):
    total_notas:      int
    total_valor_mes:  float
    notas_validadas:  int
    notas_pendentes:  int


# ─────────────────────────────────────────────────────
# AUTENTICAÇÃO (JWT)
# ─────────────────────────────────────────────────────
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2  = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


def hash_password(plain: str) -> str:
    return pwd_ctx.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)


def create_token(data: dict) -> str:
    payload = {**data, "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRE_HOURS)}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2), db: Session = Depends(get_db)) -> User:
    exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token inválido ou expirado",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if not email:
            raise exc
    except JWTError:
        raise exc
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise exc
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Acesso restrito a administradores")
    return user


# ─────────────────────────────────────────────────────
# SERVIÇO OCR — Tesseract + OpenCV
# ─────────────────────────────────────────────────────
def _deskew(gray: "np.ndarray") -> "np.ndarray":
    """Corrige inclinação usando transformada de Hough."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    if lines is None:
        return gray
    angles = []
    for _, theta in lines[:, 0]:
        angle = (theta - np.pi / 2) * 180 / np.pi
        if abs(angle) < 45:
            angles.append(angle)
    if not angles:
        return gray
    median = float(np.median(angles))
    if abs(median) < 0.5:
        return gray
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), median, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def preprocess_image(img_bytes: bytes) -> "np.ndarray":
    """Pipeline OpenCV: resize → cinza → denoise → deskew → threshold."""
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Não foi possível decodificar a imagem. Verifique o formato do arquivo.")
    # Garante largura mínima de 1200 px para melhor OCR
    h, w = img.shape[:2]
    if w < 1200:
        scale = 1200 / w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    gray = _deskew(gray)
    # Threshold adaptativo — lida melhor com variações de iluminação
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return binary


def run_ocr(img_bytes: bytes) -> Dict[str, Any]:
    """Executa OCR e retorna texto + confiança média."""
    if not OCR_AVAILABLE:
        return {"text": "", "confidence": 0, "error": "OCR não disponível no servidor"}
    processed = preprocess_image(img_bytes)
    pil_img   = Image.fromarray(processed)
    config    = "--psm 6 --oem 3"
    data = pytesseract.image_to_data(
        pil_img, lang="por", config=config, output_type=pytesseract.Output.DICT
    )
    confs = [int(c) for c in data["conf"] if int(c) > 0]
    avg   = sum(confs) / len(confs) if confs else 0
    text  = pytesseract.image_to_string(pil_img, lang="por", config=config)
    return {"text": text, "confidence": avg}


# ─────────────────────────────────────────────────────
# SERVIÇO PARSER — Regex sobre texto OCR
# ─────────────────────────────────────────────────────
def _to_float(s: str) -> Optional[float]:
    if not s:
        return None
    s = re.sub(r"[^\d,.]", "", s)
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _detect_tipo(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("nota fiscal de serviço", "nfs-e", " iss ")):
        return "nfse"
    if any(k in t for k in ("nfc-e", "cupom fiscal eletrônico", "danfe nfc")):
        return "nfce"
    return "nfe"


def parse_text(text: str) -> Dict[str, Any]:
    """Extrai campos estruturados de texto OCR via regex."""
    f: Dict[str, Any] = {}

    # CNPJ  (14 dígitos com pontuação)
    cnpjs = re.findall(r"\d{2}[\.\s]?\d{3}[\.\s]?\d{3}[/\.\s]?\d{4}[-\.\s]?\d{2}", text)
    cnpjs_clean = [re.sub(r"\D", "", c) for c in cnpjs if len(re.sub(r"\D", "", c)) == 14]
    if cnpjs_clean:
        f["emitente_cnpj"] = cnpjs_clean[0]
    if len(cnpjs_clean) > 1:
        f["destinatario_cpf_cnpj"] = cnpjs_clean[1]

    # CPF (11 dígitos) — só se destinatário ainda não preenchido
    if "destinatario_cpf_cnpj" not in f:
        cpfs = re.findall(r"\d{3}[\.\s]?\d{3}[\.\s]?\d{3}[-\.\s]?\d{2}", text)
        cpfs_clean = [re.sub(r"\D", "", c) for c in cpfs if len(re.sub(r"\D", "", c)) == 11]
        if cpfs_clean:
            f["destinatario_cpf_cnpj"] = cpfs_clean[0]

    # Chave de acesso (44 dígitos, possivelmente com espaços)
    chave_m = re.search(r"(\d[\d\s]{42,55}\d)", text)
    if chave_m:
        chave = re.sub(r"\s", "", chave_m.group(1))
        if len(chave) == 44:
            f["chave_acesso"] = chave

    # Número da nota
    num_m = re.search(r"[Nn][°ºo\.]\s*(\d{1,9})", text) or \
            re.search(r"[Nn][úu]mero\s*[:\s]\s*(\d+)", text)
    if num_m:
        f["numero"] = num_m.group(1)

    # Série
    serie_m = re.search(r"[Ss][eé]rie\s*[:\s]*(\d+)", text)
    if serie_m:
        f["serie"] = serie_m.group(1)

    # Data de emissão
    dates = re.findall(r"(\d{2}/\d{2}/\d{4})", text)
    if dates:
        try:
            f["data_emissao"] = datetime.strptime(dates[0], "%d/%m/%Y").isoformat()
        except ValueError:
            pass

    # Razão social do emitente
    razao_m = re.search(
        r"(?:Raz[aã]o\s*Social|Emitente)\s*[:\s]*([A-ZÁÀÂÃÉÊÍÓÔÕÚ][^\n]{3,60})", text
    )
    if razao_m:
        f["emitente_razao_social"] = razao_m.group(1).strip()

    # Destinatário
    dest_m = re.search(
        r"(?:Destinat[aá]rio)\s*[:\s]*([A-ZÁÀÂÃÉÊÍÓÔÕÚ][^\n]{3,60})", text, re.IGNORECASE
    )
    if dest_m:
        f["destinatario_nome"] = dest_m.group(1).strip()

    # Valor total
    for pat in (
        r"[Vv]alor\s*[Tt]otal\s*[Nn][Ff][^\d]*(\d[\d.,]+)",
        r"[Tt]otal\s*[Gg]eral[^\d]*(\d[\d.,]+)",
        r"[Vv]alor\s*[Tt]otal\s*[:\s]*R?\$?\s*(\d[\d.,]+)",
        r"[Tt][Oo][Tt][Aa][Ll][^\d]{0,20}(\d+[.,]\d{2})",
    ):
        m = re.search(pat, text)
        if m:
            f["valor_total"] = _to_float(m.group(1))
            break

    # Desconto e frete
    desc_m = re.search(r"[Dd]esconto[^\d]*(\d[\d.,]+)", text)
    if desc_m:
        f["valor_desconto"] = _to_float(desc_m.group(1))

    frete_m = re.search(r"[Ff]rete[^\d]*(\d[\d.,]+)", text)
    if frete_m:
        f["valor_frete"] = _to_float(frete_m.group(1))

    # Itens via NCM (8 dígitos seguidos de descrição)
    itens = []
    for m in re.finditer(r"(\d{8})\s+([A-Z][^\n]{5,50})", text):
        itens.append({"ncm": m.group(1), "descricao": m.group(2).strip()})
    f["itens"] = itens

    f["tipo"]    = _detect_tipo(text)
    f["tributos"] = []
    return f


# ─────────────────────────────────────────────────────
# SERVIÇO XML — Parser de NF-e
# ─────────────────────────────────────────────────────
def _strip_ns(tag: str) -> str:
    return re.sub(r"\{[^}]+\}", "", tag)


def _find(elem, *path_parts):
    """Navega pelo XML ignorando namespaces."""
    cur = elem
    for part in path_parts:
        found = next((c for c in cur if _strip_ns(c.tag) == part), None)
        if found is None:
            return None
        cur = found
    return cur


def _txt(elem, *path, default=None):
    node = _find(elem, *path)
    return node.text if node is not None else default


def parse_xml(xml_bytes: bytes) -> Dict[str, Any]:
    """Parse completo de NF-e/NFC-e XML com precisão de 100%."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        raise ValueError(f"XML inválido: {e}")

    # Localiza infNFe independentemente do namespace
    inf = next(
        (e for e in root.iter() if _strip_ns(e.tag) in ("infNFe", "infNFCe")), None
    )
    if inf is None:
        raise ValueError("Estrutura NF-e não encontrada no XML.")

    ide   = _find(inf, "ide")
    emit  = _find(inf, "emit")
    dest  = _find(inf, "dest")
    total = _find(inf, "total")
    ictot = _find(total, "ICMSTot") if total else None

    mod  = _txt(ide, "mod", default="55") if ide else "55"
    tipo = "nfce" if mod == "65" else "nfe"

    chave_raw = inf.get("Id", "")
    chave = re.sub(r"\D", "", chave_raw)
    chave = chave if len(chave) == 44 else None

    # Data de emissão
    dh = _txt(ide, "dhEmi") if ide else None
    data_emissao = None
    if dh:
        try:
            data_emissao = datetime.fromisoformat(dh.replace("Z", "+00:00")).isoformat()
        except ValueError:
            pass

    # Itens
    itens = []
    for det in inf.iter():
        if _strip_ns(det.tag) == "det":
            prod = _find(det, "prod")
            if prod:
                itens.append({
                    "descricao":      _txt(prod, "xProd"),
                    "ncm":            _txt(prod, "NCM"),
                    "quantidade":     float(_txt(prod, "qCom",  default="0")),
                    "unidade":        _txt(prod, "uCom"),
                    "valor_unitario": float(_txt(prod, "vUnCom", default="0")),
                    "valor_total":    float(_txt(prod, "vProd",  default="0")),
                    "cfop":           _txt(prod, "CFOP"),
                    "categoria":      None,
                })

    # Tributos
    tributos = []
    if ictot:
        for t_tipo, t_key in (("ICMS", "vICMS"), ("PIS", "vPIS"), ("COFINS", "vCOFINS")):
            val = _txt(ictot, t_key)
            if val and float(val) > 0:
                tributos.append({"tipo": t_tipo, "valor": float(val), "base_calculo": None, "aliquota": None})

    return {
        "tipo":                  tipo,
        "numero":                _txt(ide, "nNF") if ide else None,
        "serie":                 _txt(ide, "serie") if ide else None,
        "chave_acesso":          chave,
        "data_emissao":          data_emissao,
        "emitente_cnpj":         _txt(emit, "CNPJ") if emit else None,
        "emitente_razao_social": _txt(emit, "xNome") if emit else None,
        "destinatario_cpf_cnpj": (_txt(dest, "CNPJ") or _txt(dest, "CPF")) if dest else None,
        "destinatario_nome":     _txt(dest, "xNome") if dest else None,
        "valor_total":           float(_txt(ictot, "vNF",    default="0")) if ictot else None,
        "valor_desconto":        float(_txt(ictot, "vDesc",  default="0")) if ictot else None,
        "valor_frete":           float(_txt(ictot, "vFrete", default="0")) if ictot else None,
        "confidence_score":      100.0,
        "itens":                 itens,
        "tributos":              tributos,
    }


# ─────────────────────────────────────────────────────
# SERVIÇO SEFAZ
# ─────────────────────────────────────────────────────
async def validate_sefaz(chave: str) -> str:
    """
    Consulta simplificada de situação na SEFAZ.
    Retorna "validado", "nao_validado" ou "falha" (timeout/erro).
    Nota: a consulta oficial requer certificado digital A1/A3.
    Esta implementação usa o portal público de consulta.
    """
    if not chave or len(chave) != 44:
        return "nao_validado"
    url = (
        "https://www.nfe.fazenda.gov.br/portal/consultaRecaptcha.aspx"
        f"?tipoConsulta=resumo&tipoConteudo=7PhJ+gAVw2g=&nfe={chave}"
    )
    try:
        async with httpx.AsyncClient(timeout=SEFAZ_TIMEOUT, follow_redirects=True) as client:
            r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if "Autorizado" in r.text or "100-1 " in r.text:
            return "validado"
        return "nao_validado"
    except (httpx.TimeoutException, httpx.RequestError) as e:
        logger.warning(f"SEFAZ timeout/erro: {e}")
        return "falha"


# ─────────────────────────────────────────────────────
# FRONTEND — React 18 via CDN (sem Vite, sem build)
# ─────────────────────────────────────────────────────
FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Leitor de Nota Fiscal</title>
  <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
  <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    *{box-sizing:border-box}
    body{-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale}
    .spin{animation:spin .9s linear infinite}
    @keyframes spin{to{transform:rotate(360deg)}}
    .fade-in{animation:fade .3s ease}
    @keyframes fade{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
    input,select,button{font-family:inherit}
    ::-webkit-scrollbar{width:6px;height:6px}
    ::-webkit-scrollbar-track{background:#f1f5f9}
    ::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:3px}
  </style>
</head>
<body class="bg-gray-50 min-h-screen text-gray-800">
<div id="root"></div>
<script type="text/babel">
const {useState,useEffect,useRef,useCallback}=React;

/* ── Token em memória (nunca localStorage) ── */
let _token=null, _role=null;

async function api(method,path,body=null,isForm=false){
  const h={};
  if(_token) h["Authorization"]=`Bearer ${_token}`;
  if(body&&!isForm) h["Content-Type"]="application/json";
  const r=await fetch(path,{
    method,headers:h,
    body:body?(isForm?body:JSON.stringify(body)):undefined
  });
  if(!r.ok){
    let msg="Erro desconhecido";
    try{const e=await r.json();msg=e.detail||msg;}catch(_){}
    throw new Error(msg);
  }
  return r;
}

/* ── Ícones SVG inline ── */
const IcoUpload=()=><svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/></svg>;
const IcoOk  =()=><svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7"/></svg>;
const IcoX   =()=><svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12"/></svg>;
const IcoDown=()=><svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/></svg>;

/* ── Spinner ── */
function Spinner({text="Processando..."}){
  return(
    <div className="flex flex-col items-center justify-center py-16 fade-in">
      <div className="spin w-11 h-11 border-4 border-emerald-500 border-t-transparent rounded-full mb-4"/>
      <p className="text-gray-500 font-medium">{text}</p>
    </div>
  );
}

/* ── Navbar ── */
function Navbar({page,go,onLogout}){
  const btn=(p,label)=>(
    <button onClick={()=>go(p)}
      className={`px-4 py-1.5 rounded-md text-sm font-medium transition
        ${page===p?"bg-emerald-500 text-white shadow-sm":"text-gray-300 hover:bg-gray-700"}`}>
      {label}
    </button>
  );
  return(
    <nav className="bg-gray-900 shadow-lg sticky top-0 z-50">
      <div className="max-w-5xl mx-auto px-6 py-3.5 flex items-center justify-between">
        <span className="font-bold text-lg tracking-tight flex items-center gap-2">
          <span className="text-emerald-400">📄</span>
          <span className="text-white">Nota Fiscal</span>
        </span>
        <div className="flex items-center gap-2">
          {btn("upload","Importar")}
          {btn("dashboard","Dashboard")}
          <span className="text-gray-600 mx-2 text-xs">|</span>
          <span className="text-gray-400 text-xs capitalize">{_role}</span>
          <button onClick={onLogout} className="text-gray-400 hover:text-white text-sm ml-1 transition">Sair</button>
        </div>
      </div>
    </nav>
  );
}

/* ── Página: Login ── */
function LoginPage({onLogin}){
  const[email,setEmail]=useState("");
  const[pw,setPw]=useState("");
  const[err,setErr]=useState("");
  const[load,setLoad]=useState(false);

  async function submit(e){
    e.preventDefault();setLoad(true);setErr("");
    try{
      const fd=new FormData();fd.append("username",email);fd.append("password",pw);
      const r=await api("POST","/api/auth/token",fd,true);
      const d=await r.json();
      _token=d.access_token;_role=d.role;
      onLogin(d.role);
    }catch(ex){setErr(ex.message);}
    finally{setLoad(false);}
  }

  return(
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 via-emerald-50 to-teal-50 px-4">
      <div className="bg-white rounded-2xl shadow-xl border-t-4 border-emerald-500 p-10 w-full max-w-md fade-in">
        <div className="text-center mb-10">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-emerald-50 rounded-2xl text-3xl mb-4">📄</div>
          <h1 className="text-2xl font-bold text-gray-800">Leitor de Nota Fiscal</h1>
          <p className="text-gray-400 text-sm mt-1.5">Acesso seguro à sua equipe</p>
        </div>
        {err&&<div className="mb-5 p-3.5 bg-red-50 border border-red-200 rounded-xl text-red-600 text-sm flex gap-2 items-center"><IcoX/>{err}</div>}
        <form onSubmit={submit} className="space-y-5">
          <div>
            <label className="block text-xs font-semibold text-gray-500 mb-1.5 uppercase tracking-wider">Email</label>
            <input type="email" value={email} onChange={e=>setEmail(e.target.value)} required
              className="w-full px-4 py-3 border border-gray-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-emerald-400 focus:border-emerald-400 transition"
              placeholder="seu@empresa.com"/>
          </div>
          <div>
            <label className="block text-xs font-semibold text-gray-500 mb-1.5 uppercase tracking-wider">Senha</label>
            <input type="password" value={pw} onChange={e=>setPw(e.target.value)} required
              className="w-full px-4 py-3 border border-gray-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-emerald-400 focus:border-emerald-400 transition"
              placeholder="••••••••"/>
          </div>
          <button type="submit" disabled={load}
            className="w-full py-3 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl font-semibold text-sm transition disabled:opacity-50 mt-2">
            {load?"Entrando...":"Entrar"}
          </button>
        </form>
      </div>
    </div>
  );
}

/* ── Página: Upload ── */
function UploadPage({onReview}){
  const[drag,setDrag]=useState(false);
  const[load,setLoad]=useState(false);
  const[loadTxt,setLoadTxt]=useState("Lendo nota...");
  const[err,setErr]=useState(null);
  const ref=useRef();

  async function process(file){
    setLoad(true);setErr(null);setLoadTxt("Lendo nota...");
    const t1=setTimeout(()=>setLoadTxt("Aplicando OCR e pré-processamento..."),1800);
    const t2=setTimeout(()=>setLoadTxt("Validando na SEFAZ..."),4500);
    try{
      const fd=new FormData();fd.append("file",file);
      const r=await api("POST","/api/notas/upload",fd,true);
      const d=await r.json();
      onReview(d);
    }catch(ex){setErr(ex.message);}
    finally{clearTimeout(t1);clearTimeout(t2);setLoad(false);}
  }

  if(load) return<div className="max-w-2xl mx-auto mt-20"><Spinner text={loadTxt}/></div>;

  return(
    <div className="max-w-2xl mx-auto px-6 py-20 fade-in">
      <h2 className="text-3xl font-bold mb-2">Importar Nota Fiscal</h2>
      <p className="text-gray-400 text-sm mb-10">Suporte para NF-e XML · NFC-e · NFS-e (imagem JPG/PNG)</p>

      {err&&(
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl">
          <div className="flex gap-3">
            <div className="text-red-400 mt-0.5 flex-shrink-0"><IcoX/></div>
            <div>
              <p className="font-semibold text-red-700 text-sm">Não foi possível processar</p>
              <p className="text-red-500 text-sm mt-0.5">{err}</p>
              <div className="mt-3 bg-red-100 rounded-lg p-3 text-xs text-red-600 space-y-1">
                <p className="font-semibold">💡 Dicas para a foto:</p>
                <p>• Boa iluminação, sem sombras nem reflexos</p>
                <p>• Câmera paralela ao papel (sem ângulo)</p>
                <p>• Todos os campos da nota visíveis</p>
                <p>• Resolução mínima de 800 × 600 px</p>
              </div>
            </div>
          </div>
        </div>
      )}

      <div
        onDragOver={e=>{e.preventDefault();setDrag(true);}}
        onDragLeave={()=>setDrag(false)}
        onDrop={e=>{e.preventDefault();setDrag(false);const f=e.dataTransfer.files[0];if(f)process(f);}}
        onClick={()=>ref.current?.click()}
        className={`border-2 border-dashed rounded-2xl p-20 text-center cursor-pointer transition-all select-none
          ${drag?"border-emerald-500 bg-emerald-50 scale-[1.01]":"border-gray-200 hover:border-emerald-400 hover:bg-emerald-50/30"}`}>
        <div className={`flex justify-center mb-4 transition-colors ${drag?"text-emerald-400":"text-gray-300"}`}><IcoUpload/></div>
        <p className="font-semibold text-gray-700 text-lg">Arraste o arquivo aqui</p>
        <p className="text-gray-400 text-sm mt-1.5">ou clique para selecionar</p>
        <div className="flex justify-center gap-2 mt-4">
          {["XML","JPG","PNG","PDF"].map(t=>(
            <span key={t} className="px-2 py-0.5 bg-gray-100 text-gray-500 rounded text-xs font-mono">{t}</span>
          ))}
        </div>
      </div>
      <input ref={ref} type="file" accept=".xml,.jpg,.jpeg,.png,.pdf" className="hidden" onChange={e=>{const f=e.target.files[0];if(f)process(f);}}/>
      <p className="text-center text-gray-300 text-xs mt-6">🔒 Processamento 100% local — nenhuma imagem enviada a terceiros</p>
    </div>
  );
}

/* ── Campo de formulário reutilizável ── */
function Field({label,value,onChange,type="text",highlight,mono,options}){
  const base=`w-full px-3 py-2.5 border rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-emerald-400 focus:border-emerald-400 transition
    ${highlight?"border-yellow-300 bg-yellow-50":"border-gray-200 bg-white"}
    ${mono?"font-mono text-xs":""}`;
  return(
    <div>
      <label className="block text-xs font-semibold text-gray-500 mb-1.5 uppercase tracking-wider">{label}</label>
      {options
        ?<select value={value||""} onChange={e=>onChange(e.target.value)} className={base}>
            {options.map(o=><option key={o} value={o}>{o.toUpperCase()}</option>)}
          </select>
        :<input type={type} value={value??""} onChange={e=>onChange(e.target.value)} className={base}/>
      }
    </div>
  );
}

/* ── Página: Revisão ── */
function ReviewPage({nota,onSave,onCancel}){
  const[form,setForm]=useState({...nota});
  const[saving,setSaving]=useState(false);
  const[err,setErr]=useState("");

  function upd(k,v){setForm(f=>({...f,[k]:v}));}

  async function save(){
    setSaving(true);setErr("");
    try{
      await api("POST","/api/notas/confirm",form);
      onSave();
    }catch(ex){setErr(ex.message);setSaving(false);}
  }

  const conf=nota.confidence_score||0;
  const lowConf=conf>0&&conf<75;

  return(
    <div className="max-w-3xl mx-auto px-6 py-12 fade-in">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-3xl font-bold">Revisar Nota Fiscal</h2>
        <span className={`px-3 py-1 rounded-full text-xs font-semibold
          ${conf===100?"bg-green-100 text-green-700":conf>0?"bg-yellow-100 text-yellow-700":"bg-emerald-100 text-emerald-700"}`}>
          {conf===100?"✓ XML · 100%":conf>0?`OCR · ${conf.toFixed(0)}% confiança`:"XML"}
        </span>
      </div>

      {lowConf&&(
        <div className="mb-5 p-4 bg-yellow-50 border border-yellow-200 rounded-xl text-yellow-700 text-sm">
          ⚠️ Confiança OCR baixa — verifique os campos em amarelo antes de salvar.
        </div>
      )}
      {err&&<div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-600 text-sm">{err}</div>}

      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8 space-y-6">
        {/* Identificação */}
        <div className="grid grid-cols-2 gap-5">
          <Field label="Tipo" value={form.tipo} onChange={v=>upd("tipo",v)} options={["nfe","nfce","nfse"]}/>
          <Field label="Número" value={form.numero} onChange={v=>upd("numero",v)} highlight={lowConf}/>
        </div>
        <div className="grid grid-cols-2 gap-5">
          <Field label="Série" value={form.serie} onChange={v=>upd("serie",v)} highlight={lowConf}/>
          <Field label="Data de Emissão" value={form.data_emissao?form.data_emissao.substring(0,10):""} onChange={v=>upd("data_emissao",v)} type="date"/>
        </div>
        <Field label="Chave de Acesso (44 dígitos)" value={form.chave_acesso} onChange={v=>upd("chave_acesso",v)} highlight={lowConf} mono/>

        <hr className="border-gray-100"/>
        {/* Emitente */}
        <p className="text-xs font-bold text-emerald-600 uppercase tracking-widest border-l-2 border-emerald-400 pl-3">Emitente</p>
        <div className="grid grid-cols-2 gap-5">
          <Field label="CNPJ" value={form.emitente_cnpj} onChange={v=>upd("emitente_cnpj",v)} highlight={lowConf} mono/>
          <Field label="Razão Social" value={form.emitente_razao_social} onChange={v=>upd("emitente_razao_social",v)} highlight={lowConf}/>
        </div>

        <hr className="border-gray-100"/>
        {/* Destinatário */}
        <p className="text-xs font-bold text-emerald-600 uppercase tracking-widest border-l-2 border-emerald-400 pl-3">Destinatário</p>
        <div className="grid grid-cols-2 gap-5">
          <Field label="CPF / CNPJ" value={form.destinatario_cpf_cnpj} onChange={v=>upd("destinatario_cpf_cnpj",v)} highlight={lowConf} mono/>
          <Field label="Nome" value={form.destinatario_nome} onChange={v=>upd("destinatario_nome",v)} highlight={lowConf}/>
        </div>

        <hr className="border-gray-100"/>
        {/* Valores */}
        <p className="text-xs font-bold text-emerald-600 uppercase tracking-widest border-l-2 border-emerald-400 pl-3">Valores (R$)</p>
        <div className="grid grid-cols-3 gap-4">
          <Field label="Total" value={form.valor_total??""} onChange={v=>upd("valor_total",parseFloat(v)||null)} type="number" highlight={lowConf}/>
          <Field label="Desconto" value={form.valor_desconto??""} onChange={v=>upd("valor_desconto",parseFloat(v)||0)} type="number"/>
          <Field label="Frete" value={form.valor_frete??""} onChange={v=>upd("valor_frete",parseFloat(v)||0)} type="number"/>
        </div>

        {/* Itens */}
        {form.itens&&form.itens.length>0&&(
          <>
            <hr className="border-gray-100"/>
            <p className="text-xs font-bold text-emerald-600 uppercase tracking-widest border-l-2 border-emerald-400 pl-3">Itens ({form.itens.length})</p>
            <div className="overflow-x-auto rounded-lg border border-gray-100">
              <table className="w-full text-sm">
                <thead className="bg-gray-50 text-gray-400 text-xs uppercase">
                  <tr>
                    <th className="px-3 py-2 text-left">Descrição</th>
                    <th className="px-3 py-2 text-left">NCM</th>
                    <th className="px-3 py-2 text-right">Qtd</th>
                    <th className="px-3 py-2 text-right">Unit.</th>
                    <th className="px-3 py-2 text-right">Total</th>
                  </tr>
                </thead>
                <tbody>
                  {form.itens.map((it,i)=>(
                    <tr key={i} className="border-t border-gray-50">
                      <td className="px-3 py-2 max-w-[200px] truncate">{it.descricao||"—"}</td>
                      <td className="px-3 py-2 font-mono text-xs text-gray-500">{it.ncm||"—"}</td>
                      <td className="px-3 py-2 text-right">{it.quantidade??""} {it.unidade||""}</td>
                      <td className="px-3 py-2 text-right">{it.valor_unitario!=null?`R$\u00a0${it.valor_unitario.toFixed(2)}`:"—"}</td>
                      <td className="px-3 py-2 text-right font-medium">{it.valor_total!=null?`R$\u00a0${it.valor_total.toFixed(2)}`:"—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>

      <div className="flex gap-3 mt-8">
        <button onClick={onCancel} className="px-6 py-3 border border-gray-200 rounded-xl text-sm font-medium hover:bg-gray-50 transition">Cancelar</button>
        <button onClick={save} disabled={saving}
          className="flex-1 py-3 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition disabled:opacity-50">
          {saving
            ?<><span className="spin w-4 h-4 border-2 border-white border-t-transparent rounded-full"/>Salvando...</>
            :<><IcoOk/>Confirmar e Salvar</>
          }
        </button>
      </div>
    </div>
  );
}

/* ── Badge de status SEFAZ ── */
function StatusBadge({s}){
  const map={validado:"bg-green-100 text-green-700",nao_validado:"bg-gray-100 text-gray-500",falha:"bg-red-100 text-red-600"};
  const lbl={validado:"Validado",nao_validado:"Não validado",falha:"Falha SEFAZ"};
  return<span className={`px-2 py-0.5 rounded-full text-xs font-medium ${map[s]||map.nao_validado}`}>{lbl[s]||s}</span>;
}

/* ── Card de estatística ── */
function StatCard({label,value,accent}){
  const acc={
    emerald:"border-l-emerald-500 text-emerald-700 bg-emerald-50/60",
    green:  "border-l-green-500 text-green-700 bg-green-50/60",
    teal:   "border-l-teal-500 text-teal-700 bg-teal-50/60",
    rose:   "border-l-rose-500 text-rose-700 bg-rose-50/60",
  };
  return(
    <div className={`rounded-xl p-6 border-l-4 shadow-sm ${acc[accent]||acc.emerald}`}>
      <p className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-2">{label}</p>
      <p className="text-3xl font-bold">{value}</p>
    </div>
  );
}

/* ── Página: Dashboard ── */
function DashboardPage(){
  const[notas,setNotas]=useState([]);
  const[stats,setStats]=useState(null);
  const[load,setLoad]=useState(true);
  const[fSearch,setFSearch]=useState("");
  const[fTipo,setFTipo]=useState("");
  const[fStatus,setFStatus]=useState("");

  useEffect(()=>{fetchAll();},[]);

  async function fetchAll(){
    setLoad(true);
    try{
      const[rN,rS]=await Promise.all([api("GET","/api/notas"),api("GET","/api/dashboard/stats")]);
      setNotas(await rN.json());
      setStats(await rS.json());
    }catch(ex){console.error(ex);}
    finally{setLoad(false);}
  }

  async function exportCsv(){
    try{
      const r=await api("GET","/api/dashboard/export/csv");
      const blob=await r.blob();
      const url=URL.createObjectURL(blob);
      const a=document.createElement("a");
      a.href=url;a.download=`notas_${new Date().toISOString().slice(0,10)}.csv`;
      a.click();URL.revokeObjectURL(url);
    }catch(ex){alert("Erro ao exportar: "+ex.message);}
  }

  async function del(id){
    if(!confirm("Excluir esta nota fiscal permanentemente?"))return;
    try{await api("DELETE",`/api/notas/${id}`);setNotas(n=>n.filter(x=>x.id!==id));}
    catch(ex){alert(ex.message);}
  }

  const filtered=notas.filter(n=>{
    if(fTipo&&n.tipo!==fTipo)return false;
    if(fStatus&&n.status_sefaz!==fStatus)return false;
    if(fSearch){
      const s=fSearch.toLowerCase();
      return(n.emitente_razao_social||"").toLowerCase().includes(s)||
             (n.numero||"").includes(s)||(n.emitente_cnpj||"").includes(s);
    }
    return true;
  });

  const fmt=(v)=>v!=null?`R$\u00a0${Number(v).toFixed(2)}`:"—";

  if(load)return<div className="max-w-5xl mx-auto mt-10"><Spinner text="Carregando..."/></div>;

  return(
    <div className="max-w-5xl mx-auto px-6 py-12 fade-in">
      <div className="flex items-center justify-between mb-8">
        <h2 className="text-3xl font-bold">Dashboard</h2>
        <button onClick={exportCsv}
          className="flex items-center gap-2 px-5 py-2.5 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl text-sm font-semibold transition shadow-sm">
          <IcoDown/>Exportar CSV
        </button>
      </div>

      {stats&&(
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-5 mb-10">
          <StatCard label="Total de Notas"   value={stats.total_notas}              accent="emerald"/>
          <StatCard label="Valor do Mês"     value={fmt(stats.total_valor_mes)}     accent="green"/>
          <StatCard label="Validadas SEFAZ"  value={stats.notas_validadas}          accent="teal"/>
          <StatCard label="Pendentes"        value={stats.notas_pendentes}          accent="rose"/>
        </div>
      )}

      {/* Filtros */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
        <div className="p-5 border-b border-gray-100 flex flex-wrap gap-3">
          <input type="text" placeholder="Buscar emitente, número…" value={fSearch} onChange={e=>setFSearch(e.target.value)}
            className="flex-1 min-w-[200px] px-4 py-2.5 border border-gray-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-emerald-400 focus:border-emerald-400 transition"/>
          <select value={fTipo} onChange={e=>setFTipo(e.target.value)}
            className="px-4 py-2.5 border border-gray-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-emerald-400 focus:border-emerald-400 transition">
            <option value="">Todos os tipos</option>
            <option value="nfe">NF-e</option><option value="nfce">NFC-e</option><option value="nfse">NFS-e</option>
          </select>
          <select value={fStatus} onChange={e=>setFStatus(e.target.value)}
            className="px-4 py-2.5 border border-gray-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-emerald-400 focus:border-emerald-400 transition">
            <option value="">Todos os status</option>
            <option value="validado">Validado</option>
            <option value="nao_validado">Não validado</option>
            <option value="falha">Falha</option>
          </select>
        </div>

        {/* Tabela */}
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 text-gray-400 text-xs uppercase border-b">
              <tr>
                <th className="px-5 py-4 text-left">Tipo</th>
                <th className="px-5 py-4 text-left">Nº</th>
                <th className="px-5 py-4 text-left">Emitente</th>
                <th className="px-5 py-4 text-left">Emissão</th>
                <th className="px-5 py-4 text-right">Valor Total</th>
                <th className="px-5 py-4 text-center">SEFAZ</th>
                {_role==="admin"&&<th className="px-5 py-4 text-center">Ação</th>}
              </tr>
            </thead>
            <tbody>
              {filtered.length===0
                ?<tr><td colSpan={_role==="admin"?7:6} className="text-center py-16 text-gray-300">Nenhuma nota encontrada</td></tr>
                :filtered.map(n=>(
                  <tr key={n.id} className="border-t border-gray-50 hover:bg-emerald-50/30 transition">
                    <td className="px-5 py-4">
                      <span className="px-2.5 py-1 bg-emerald-100 text-emerald-700 rounded-lg text-xs font-semibold uppercase">{n.tipo}</span>
                    </td>
                    <td className="px-5 py-4 font-mono text-xs text-gray-500">{n.numero||"—"}</td>
                    <td className="px-5 py-4 max-w-[200px] truncate">{n.emitente_razao_social||n.emitente_cnpj||"—"}</td>
                    <td className="px-5 py-4 text-gray-400 text-xs">{n.data_emissao?new Date(n.data_emissao).toLocaleDateString("pt-BR"):"—"}</td>
                    <td className="px-5 py-4 text-right font-semibold">{fmt(n.valor_total)}</td>
                    <td className="px-5 py-4 text-center"><StatusBadge s={n.status_sefaz}/></td>
                    {_role==="admin"&&(
                      <td className="px-5 py-4 text-center">
                        <button onClick={()=>del(n.id)} className="text-gray-300 hover:text-red-500 transition"><IcoX/></button>
                      </td>
                    )}
                  </tr>
                ))
              }
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

/* ── App Root ── */
function App(){
  const[page,setPage]=useState("login");
  const[,setRole]=useState(null);
  const[pendingNota,setPendingNota]=useState(null);

  function go(p){setPage(p);}

  function onLogin(r){setRole(r);go("upload");}
  function onLogout(){_token=null;_role=null;setRole(null);go("login");}
  function onReview(n){setPendingNota(n);go("review");}
  function onSaved(){setPendingNota(null);go("dashboard");}

  if(page==="login")return<LoginPage onLogin={onLogin}/>;

  return(
    <div>
      <Navbar page={page} go={go} onLogout={onLogout}/>
      {page==="upload"   &&<UploadPage    onReview={onReview}/>}
      {page==="review"   &&<ReviewPage    nota={pendingNota} onSave={onSaved} onCancel={()=>go("upload")}/>}
      {page==="dashboard"&&<DashboardPage/>}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App/>);
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────
# FASTAPI — APP + LIFESPAN
# ─────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cria tabelas no banco (se não existirem)
    Base.metadata.create_all(bind=engine)
    # Cria usuário admin padrão se banco vazio
    db = SessionLocal()
    try:
        if not db.query(User).first():
            db.add(User(
                email="admin@empresa.com",
                hashed_password=hash_password("admin123"),
                role="admin",
            ))
            db.commit()
            logger.info("Usuário padrão criado: admin@empresa.com / admin123")
    finally:
        db.close()
    yield


app = FastAPI(title="Nota Fiscal Reader", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────
# ROTA — Frontend
# ─────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def frontend():
    return FRONTEND_HTML


# ─────────────────────────────────────────────────────
# ROTAS — Autenticação
# ─────────────────────────────────────────────────────
@app.post("/api/auth/token", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form.username).first()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Email ou senha incorretos")
    token = create_token({"sub": user.email, "role": user.role})
    return {"access_token": token, "token_type": "bearer", "role": user.role}


@app.post("/api/auth/register", response_model=UserOut)
async def register(
    body: UserCreate,
    _: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(status_code=400, detail="Email já cadastrado")
    if body.role not in ("admin", "viewer"):
        raise HTTPException(status_code=400, detail="Role inválido. Use 'admin' ou 'viewer'")
    u = User(email=body.email, hashed_password=hash_password(body.password), role=body.role)
    db.add(u); db.commit(); db.refresh(u)
    return u


@app.get("/api/auth/users", response_model=List[UserOut])
async def list_users(_: User = Depends(require_admin), db: Session = Depends(get_db)):
    return db.query(User).all()


# ─────────────────────────────────────────────────────
# ROTAS — Notas Fiscais
# ─────────────────────────────────────────────────────
@app.post("/api/notas/upload")
async def upload_nota(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """
    Recebe arquivo (XML ou imagem), processa e retorna campos extraídos para revisão.
    Imagem nunca é persistida em disco.
    """
    content  = await file.read()
    filename = (file.filename or "").lower()

    # ── XML de NF-e → parse direto (100% preciso)
    is_xml = filename.endswith(".xml") or content[:5] in (b"<?xml", b"<nfeP", b"<NFe ", b"<nfce")
    if is_xml:
        try:
            return parse_xml(content)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    # ── Imagem → OCR
    if not OCR_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=(
                "OCR não disponível. Instale: "
                "pip install pytesseract opencv-python-headless numpy pillow"
            ),
        )
    try:
        result = run_ocr(content)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Erro ao processar imagem: {e}")

    if result["confidence"] < OCR_CONFIDENCE_THRESHOLD:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Qualidade da imagem insuficiente "
                f"(confiança OCR: {result['confidence']:.0f}% < {OCR_CONFIDENCE_THRESHOLD}%). "
                "Tire uma foto mais nítida, com boa iluminação e sem sombras."
            ),
        )

    extracted = parse_text(result["text"])
    extracted["confidence_score"] = result["confidence"]
    extracted["status_sefaz"]     = "nao_validado"
    return extracted


@app.post("/api/notas/confirm", response_model=NotaFiscalOut)
async def confirm_nota(
    body: NotaFiscalCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Salva nota confirmada pelo usuário no banco de dados."""
    if body.tipo not in ("nfe", "nfce", "nfse"):
        raise HTTPException(status_code=400, detail="Tipo inválido. Use nfe, nfce ou nfse")

    # Verifica chave duplicada
    if body.chave_acesso:
        dup = db.query(NotaFiscalDB).filter(NotaFiscalDB.chave_acesso == body.chave_acesso).first()
        if dup:
            raise HTTPException(status_code=400, detail="Nota já importada (chave de acesso duplicada)")

    # Validação SEFAZ (apenas NF-e / NFC-e com chave)
    status_sefaz = "nao_validado"
    if body.tipo in ("nfe", "nfce") and body.chave_acesso:
        status_sefaz = await validate_sefaz(body.chave_acesso)

    # Converte data_emissao
    data_emissao = None
    if body.data_emissao:
        try:
            data_emissao = datetime.fromisoformat(body.data_emissao.replace("Z", "+00:00"))
        except ValueError:
            pass

    nota = NotaFiscalDB(
        tipo=body.tipo,
        numero=body.numero,
        serie=body.serie,
        chave_acesso=body.chave_acesso,
        data_emissao=data_emissao,
        emitente_cnpj=body.emitente_cnpj,
        emitente_razao_social=body.emitente_razao_social,
        destinatario_cpf_cnpj=body.destinatario_cpf_cnpj,
        destinatario_nome=body.destinatario_nome,
        valor_total=body.valor_total,
        valor_desconto=body.valor_desconto or 0.0,
        valor_frete=body.valor_frete or 0.0,
        status_sefaz=status_sefaz,
        confidence_score=body.confidence_score,
        importado_por_id=current_user.id,
    )
    db.add(nota)
    db.flush()

    _item_keys = ["descricao", "ncm", "quantidade", "unidade", "valor_unitario", "valor_total", "cfop", "categoria"]
    for it in body.itens:
        db.add(ItemNotaDB(nota_id=nota.id, **{k: it.get(k) for k in _item_keys}))

    _trib_keys = ["tipo", "base_calculo", "aliquota", "valor"]
    for tr in body.tributos:
        db.add(TributoNotaDB(nota_id=nota.id, **{k: tr.get(k) for k in _trib_keys}))

    db.commit()
    db.refresh(nota)
    return nota


@app.get("/api/notas", response_model=List[NotaFiscalOut])
async def list_notas(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    q = db.query(NotaFiscalDB)
    if current_user.role != "admin":
        q = q.filter(NotaFiscalDB.importado_por_id == current_user.id)
    return q.order_by(NotaFiscalDB.data_importacao.desc()).all()


@app.get("/api/notas/{nota_id}", response_model=NotaFiscalOut)
async def get_nota(
    nota_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    nota = db.query(NotaFiscalDB).filter(NotaFiscalDB.id == nota_id).first()
    if not nota:
        raise HTTPException(status_code=404, detail="Nota não encontrada")
    if current_user.role != "admin" and nota.importado_por_id != current_user.id:
        raise HTTPException(status_code=403, detail="Acesso negado")
    return nota


@app.delete("/api/notas/{nota_id}")
async def delete_nota(
    nota_id: int,
    _: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    nota = db.query(NotaFiscalDB).filter(NotaFiscalDB.id == nota_id).first()
    if not nota:
        raise HTTPException(status_code=404, detail="Nota não encontrada")
    db.delete(nota)
    db.commit()
    return {"ok": True}


# ─────────────────────────────────────────────────────
# ROTAS — Dashboard
# ─────────────────────────────────────────────────────
@app.get("/api/dashboard/stats", response_model=DashboardStats)
async def dashboard_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    q = db.query(NotaFiscalDB)
    if current_user.role != "admin":
        q = q.filter(NotaFiscalDB.importado_por_id == current_user.id)
    all_n = q.all()

    now   = datetime.now(timezone.utc)
    som   = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    def _dt(n):
        if not n.data_emissao:
            return None
        dt = n.data_emissao
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    total_mes = sum(
        (n.valor_total or 0) for n in all_n
        if _dt(n) is not None and _dt(n) >= som
    )

    return DashboardStats(
        total_notas=len(all_n),
        total_valor_mes=total_mes,
        notas_validadas=sum(1 for n in all_n if n.status_sefaz == "validado"),
        notas_pendentes=sum(1 for n in all_n if n.status_sefaz != "validado"),
    )


@app.get("/api/dashboard/export/csv")
async def export_csv(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    q = db.query(NotaFiscalDB)
    if current_user.role != "admin":
        q = q.filter(NotaFiscalDB.importado_por_id == current_user.id)
    notas = q.order_by(NotaFiscalDB.data_importacao.desc()).all()

    buf = io.StringIO()
    w   = csv.writer(buf)
    w.writerow([
        "ID", "Tipo", "Número", "Série", "Chave de Acesso",
        "Data Emissão", "Data Importação",
        "Emitente CNPJ", "Emitente Razão Social",
        "Destinatário CPF/CNPJ", "Destinatário Nome",
        "Valor Total (R$)", "Valor Desconto (R$)", "Valor Frete (R$)",
        "Status SEFAZ", "Confiança OCR (%)",
    ])
    for n in notas:
        w.writerow([
            n.id, n.tipo, n.numero, n.serie, n.chave_acesso,
            n.data_emissao.isoformat()   if n.data_emissao   else "",
            n.data_importacao.isoformat() if n.data_importacao else "",
            n.emitente_cnpj, n.emitente_razao_social,
            n.destinatario_cpf_cnpj, n.destinatario_nome,
            n.valor_total, n.valor_desconto, n.valor_frete,
            n.status_sefaz, n.confidence_score,
        ])
    buf.seek(0)

    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=notas_fiscais.csv"},
    )


# ─────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    print()
    print("━" * 52)
    print("  Leitor de Nota Fiscal")
    print("━" * 52)
    print("  URL    → http://localhost:8000")
    print("  Login  → admin@empresa.com")
    print("  Senha  → admin123")
    print("  OCR    →", "disponível ✓" if OCR_AVAILABLE else "NÃO disponível (só XML)")
    print("  Banco  →", DATABASE_URL)
    print("━" * 52)
    print()

    uvicorn.run("nota_fiscal:app", host="0.0.0.0", port=8000, reload=True)
