from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime
from models.models import UserRole, DocumentType, SefazStatus


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    role: UserRole = UserRole.viewer


class UserOut(BaseModel):
    id: int
    email: str
    role: UserRole
    created_at: datetime
    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str


class ItemNotaOut(BaseModel):
    id: int
    descricao: Optional[str] = None
    ncm: Optional[str] = None
    quantidade: Optional[float] = None
    unidade: Optional[str] = None
    valor_unitario: Optional[float] = None
    valor_total: Optional[float] = None
    cfop: Optional[str] = None
    class Config:
        from_attributes = True


class NotaFiscalCreate(BaseModel):
    tipo: Optional[DocumentType] = None
    numero: Optional[str] = None
    serie: Optional[str] = None
    chave_acesso: Optional[str] = None
    data_emissao: Optional[datetime] = None
    emitente_cnpj: Optional[str] = None
    emitente_razao_social: Optional[str] = None
    destinatario_cpf_cnpj: Optional[str] = None
    destinatario_nome: Optional[str] = None
    valor_total: Optional[float] = None
    valor_desconto: Optional[float] = 0
    valor_frete: Optional[float] = 0
    observacoes: Optional[str] = None
    pix_txid: Optional[str] = None
    pix_pagador_nome: Optional[str] = None
    pix_pagador_cpf_cnpj: Optional[str] = None
    pix_recebedor_nome: Optional[str] = None
    pix_recebedor_chave: Optional[str] = None
    confirmado: Optional[bool] = False


class NotaFiscalOut(NotaFiscalCreate):
    id: int
    status_sefaz: Optional[SefazStatus] = None
    confidence_score: Optional[float] = None
    data_importacao: Optional[datetime] = None
    itens: List[ItemNotaOut] = []
    class Config:
        from_attributes = True


class DashboardStats(BaseModel):
    total_mes: float
    total_pix: float
    total_nf: float
    aguardando_revisao: int
    total_documentos: int
