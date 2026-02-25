from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Enum, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from db import Base


class UserRole(str, enum.Enum):
    admin = "admin"
    viewer = "viewer"


class DocumentType(str, enum.Enum):
    nfe = "nfe"
    nfce = "nfce"
    nfse = "nfse"
    pix = "pix"


class SefazStatus(str, enum.Enum):
    validado = "validado"
    nao_validado = "nao_validado"
    falha = "falha"
    nao_aplicavel = "nao_aplicavel"


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum(UserRole), default=UserRole.viewer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    notas = relationship("NotaFiscal", back_populates="importado_por_user")


class NotaFiscal(Base):
    __tablename__ = "notas_fiscais"
    id = Column(Integer, primary_key=True)
    tipo = Column(Enum(DocumentType), nullable=False)
    numero = Column(String)
    serie = Column(String)
    chave_acesso = Column(String)
    data_emissao = Column(DateTime(timezone=True))
    data_importacao = Column(DateTime(timezone=True), server_default=func.now())
    emitente_cnpj = Column(String)
    emitente_razao_social = Column(String)
    destinatario_cpf_cnpj = Column(String)
    destinatario_nome = Column(String)
    valor_total = Column(Float)
    valor_desconto = Column(Float, default=0)
    valor_frete = Column(Float, default=0)
    status_sefaz = Column(Enum(SefazStatus), default=SefazStatus.nao_validado)
    confidence_score = Column(Float)
    confirmado = Column(Boolean, default=False, nullable=False, server_default='0')
    observacoes = Column(String)
    importado_por = Column(Integer, ForeignKey("users.id"))
    importado_por_user = relationship("User", back_populates="notas")
    itens = relationship("ItemNota", back_populates="nota", cascade="all, delete-orphan")
    tributos = relationship("TributoNota", back_populates="nota", cascade="all, delete-orphan")
    # PIX specific
    pix_txid = Column(String)
    pix_pagador_nome = Column(String)
    pix_pagador_cpf_cnpj = Column(String)
    pix_recebedor_nome = Column(String)
    pix_recebedor_chave = Column(String)


class ItemNota(Base):
    __tablename__ = "itens_nota"
    id = Column(Integer, primary_key=True)
    nota_id = Column(Integer, ForeignKey("notas_fiscais.id"))
    descricao = Column(String)
    ncm = Column(String)
    quantidade = Column(Float)
    unidade = Column(String)
    valor_unitario = Column(Float)
    valor_total = Column(Float)
    cfop = Column(String)
    nota = relationship("NotaFiscal", back_populates="itens")


class TributoNota(Base):
    __tablename__ = "tributos_nota"
    id = Column(Integer, primary_key=True)
    nota_id = Column(Integer, ForeignKey("notas_fiscais.id"))
    tipo = Column(String)
    base_calculo = Column(Float)
    aliquota = Column(Float)
    valor = Column(Float)
    nota = relationship("NotaFiscal", back_populates="tributos")
