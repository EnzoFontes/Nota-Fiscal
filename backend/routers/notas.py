import io
import csv
import logging
import xml.etree.ElementTree as ET
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List, Optional
from db import get_db
from models.models import User, NotaFiscal, ItemNota, TributoNota
from schemas.schemas import NotaFiscalCreate, NotaFiscalOut
from services.ocr_service import extract_text_from_image
from services.parser_service import parse_nf_image, parse_pix_receipt, is_pix_receipt, is_nfce
from services.xml_service import parse_nfe_xml
from services.sefaz_service import validate_chave_acesso
from services.pix_validator import validate_pix_document, analyze_with_claude
from services.qr_service import extract_from_qr
from services.pdf_service import process_pdf
from routers.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/notas", tags=["notas"])

ALLOWED_IMAGES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/tiff"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif"}
_PDF_TYPES = {"application/pdf"}
_PDF_EXTS = {".pdf"}


def _file_ext(filename: str) -> str:
    """Return the lowercased dot-prefixed extension of a filename, e.g. '.pdf'."""
    return "." + (filename or "").rsplit(".", 1)[-1].lower()


def _is_image(content_type: str, filename: str) -> bool:
    return content_type in ALLOWED_IMAGES or _file_ext(filename) in _IMAGE_EXTS


def _is_pdf(content_type: str, filename: str) -> bool:
    return content_type in _PDF_TYPES or _file_ext(filename) in _PDF_EXTS


@router.post("/upload", response_model=NotaFiscalOut)
async def upload_document(file: UploadFile = File(...),
                           db: Session = Depends(get_db),
                           current_user: User = Depends(get_current_user)):
    content = await file.read()
    ct = (file.content_type or "").lower()
    fname = file.filename or ""

    if ct in ("application/xml", "text/xml") or fname.endswith(".xml"):
        parsed = parse_nfe_xml(content)

    elif _is_pdf(ct, fname):
        # ── PDF branch ────────────────────────────────────────────────────────
        pdf_result = process_pdf(content)
        text = pdf_result["text"]
        is_scanned = pdf_result["is_scanned"]

        if not text.strip():
            logger.warning("PDF: no text extracted. Saving blank PIX for manual entry.")
            parsed = parse_pix_receipt("")
            parsed["tipo"] = "pix"
        else:
            if is_pix_receipt(text):
                parsed = parse_pix_receipt(text)
                parsed["tipo"] = "pix"
            else:
                parsed = parse_nf_image(text)
                parsed["tipo"] = "nfce" if is_nfce(text) else "nfe"

        parsed["confidence_score"] = 100.0 if not is_scanned else pdf_result.get("confidence", 0.0)

    elif _is_image(ct, fname):
        # ── Image branch ──────────────────────────────────────────────────────
        # Step 1: QR code scan (highest priority — exact machine-readable data)
        qr_data = extract_from_qr(content)

        # Step 2: OCR text extraction
        ocr = extract_text_from_image(content)
        text = ocr["text"]

        # Step 3: Merge QR + OCR results (QR wins on overlapping fields)
        if not text.strip() and not qr_data:
            logger.warning(f"OCR empty and no QR (confidence={ocr['confidence']}). Saving blank PIX.")
            parsed = parse_pix_receipt("")
            parsed["tipo"] = "pix"
        else:
            if is_pix_receipt(text) or (qr_data and qr_data.get("tipo") == "pix"):
                parsed = parse_pix_receipt(text)
                parsed["tipo"] = "pix"
            else:
                parsed = parse_nf_image(text)
                parsed["tipo"] = "nfce" if is_nfce(text) else "nfe"

            if qr_data:
                for k, v in qr_data.items():
                    if v is not None:
                        parsed[k] = v
                logger.info(f"QR data merged: {list(qr_data.keys())}")

        parsed["confidence_score"] = ocr["confidence"]

        # Step 4: PIX validation
        if parsed.get("tipo") == "pix" and text.strip():
            validation = validate_pix_document(text, content)
            logger.info(f"PIX validation score={validation['score']} valid={validation['is_valid']} warnings={validation['warnings']}")
            if validation["warnings"]:
                logger.warning(f"PIX validation warnings: {validation['warnings']}")
            claude_analysis = analyze_with_claude(text, validation)
            if claude_analysis:
                logger.info(f"Claude analysis: {claude_analysis[:200]}")

    else:
        raise HTTPException(
            status_code=400,
            detail="Formato não suportado. Use XML, PDF ou imagem (JPG, PNG, WEBP, TIFF)",
        )

    # SEFAZ validation for NF-e
    sefaz_status = "nao_aplicavel"
    if parsed.get("tipo") == "nfe" and parsed.get("chave_acesso"):
        result = await validate_chave_acesso(parsed["chave_acesso"])
        sefaz_status = result["status"]

    itens = parsed.pop("itens", [])
    tributos = parsed.pop("tributos", [])

    allowed_fields = {c.key for c in NotaFiscal.__table__.columns}
    nota_data = {k: v for k, v in parsed.items() if k in allowed_fields}
    nota = NotaFiscal(**nota_data, status_sefaz=sefaz_status,
                      importado_por=current_user.id)
    db.add(nota)
    db.flush()

    for item in itens:
        db.add(ItemNota(nota_id=nota.id, **item))
    for trib in tributos:
        db.add(TributoNota(nota_id=nota.id, **trib))

    db.commit()
    db.refresh(nota)
    return nota


@router.post("/novo-pix", response_model=NotaFiscalOut)
async def create_manual_pix(db: Session = Depends(get_db),
                             current_user: User = Depends(get_current_user)):
    """Create a blank PIX document for manual data entry in the Review page."""
    nota = NotaFiscal(
        tipo="pix",
        status_sefaz="nao_aplicavel",
        importado_por=current_user.id,
    )
    db.add(nota)
    db.commit()
    db.refresh(nota)
    return nota


@router.get("/export/csv")
async def export_csv(tipo: Optional[str] = None,
                     data_inicio: Optional[datetime] = None,
                     data_fim: Optional[datetime] = None,
                     db: Session = Depends(get_db),
                     _: User = Depends(get_current_user)):
    q = db.query(NotaFiscal)
    if tipo:
        q = q.filter(NotaFiscal.tipo == tipo)
    if data_inicio:
        q = q.filter(NotaFiscal.data_importacao >= data_inicio)
    if data_fim:
        q = q.filter(NotaFiscal.data_importacao <= data_fim)
    notas = q.order_by(NotaFiscal.data_importacao.desc()).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Tipo", "Número", "Emitente", "Destinatário",
                     "Valor Total", "Data Emissão", "Status SEFAZ", "Data Importação"])
    for n in notas:
        writer.writerow([n.id, n.tipo, n.numero, n.emitente_razao_social,
                         n.destinatario_nome, n.valor_total, n.data_emissao,
                         n.status_sefaz, n.data_importacao])
    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8-sig")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=notas_fiscais.csv"},
    )


@router.get("/export/xml")
async def export_xml(tipo: Optional[str] = None,
                     data_inicio: Optional[datetime] = None,
                     data_fim: Optional[datetime] = None,
                     db: Session = Depends(get_db),
                     _: User = Depends(get_current_user)):
    q = db.query(NotaFiscal).filter(NotaFiscal.confirmado == True)  # noqa: E712
    if tipo:
        q = q.filter(NotaFiscal.tipo == tipo)
    if data_inicio:
        q = q.filter(NotaFiscal.data_importacao >= data_inicio)
    if data_fim:
        q = q.filter(NotaFiscal.data_importacao <= data_fim)
    notas = q.order_by(NotaFiscal.data_importacao.desc()).all()

    def _val(v) -> str:
        """Return the plain string value of enums or other types."""
        if v is None:
            return ""
        return v.value if hasattr(v, "value") else str(v)

    def _txt(parent: ET.Element, tag: str, value) -> ET.Element:
        el = ET.SubElement(parent, tag)
        el.text = _val(value)
        return el

    root = ET.Element("NotasFiscais")
    root.set("geradoEm", datetime.now().isoformat())
    root.set("totalDocumentos", str(len(notas)))

    for n in notas:
        doc = ET.SubElement(root, "NotaFiscal", id=str(n.id), tipo=_val(n.tipo))

        # Identification
        ident = ET.SubElement(doc, "Identificacao")
        _txt(ident, "Numero", n.numero)
        _txt(ident, "Serie", n.serie)
        _txt(ident, "ChaveAcesso", n.chave_acesso)
        _txt(ident, "DataEmissao", n.data_emissao.isoformat() if n.data_emissao else "")
        _txt(ident, "DataImportacao", n.data_importacao.isoformat() if n.data_importacao else "")
        _txt(ident, "StatusSEFAZ", n.status_sefaz)

        # Emitente
        emit = ET.SubElement(doc, "Emitente")
        _txt(emit, "CNPJ", n.emitente_cnpj)
        _txt(emit, "RazaoSocial", n.emitente_razao_social)

        # Destinatário
        dest = ET.SubElement(doc, "Destinatario")
        _txt(dest, "CPFCNPJ", n.destinatario_cpf_cnpj)
        _txt(dest, "Nome", n.destinatario_nome)

        # Valores
        vals = ET.SubElement(doc, "Valores")
        _txt(vals, "Total", f"{n.valor_total:.2f}" if n.valor_total is not None else "")
        _txt(vals, "Desconto", f"{n.valor_desconto:.2f}" if n.valor_desconto is not None else "")
        _txt(vals, "Frete", f"{n.valor_frete:.2f}" if n.valor_frete is not None else "")

        # PIX fields
        if n.tipo and _val(n.tipo) == "pix":
            pix = ET.SubElement(doc, "DadosPIX")
            _txt(pix, "TXID", n.pix_txid)
            _txt(pix, "PagadorNome", n.pix_pagador_nome)
            _txt(pix, "PagadorCPFCNPJ", n.pix_pagador_cpf_cnpj)
            _txt(pix, "RecebedorNome", n.pix_recebedor_nome)
            _txt(pix, "ChavePIX", n.pix_recebedor_chave)

        # Items
        if n.itens:
            itens_el = ET.SubElement(doc, "Itens")
            for item in n.itens:
                it = ET.SubElement(itens_el, "Item")
                _txt(it, "Descricao", item.descricao)
                _txt(it, "NCM", item.ncm)
                _txt(it, "CFOP", item.cfop)
                _txt(it, "Quantidade", item.quantidade)
                _txt(it, "Unidade", item.unidade)
                _txt(it, "ValorUnitario", f"{item.valor_unitario:.4f}" if item.valor_unitario is not None else "")
                _txt(it, "ValorTotal", f"{item.valor_total:.2f}" if item.valor_total is not None else "")

        # Tributos
        if n.tributos:
            trib_el = ET.SubElement(doc, "Tributos")
            for t in n.tributos:
                tr = ET.SubElement(trib_el, "Tributo", tipo=str(t.tipo or ""))
                _txt(tr, "BaseCalculo", f"{t.base_calculo:.2f}" if t.base_calculo is not None else "")
                _txt(tr, "Aliquota", f"{t.aliquota:.4f}" if t.aliquota is not None else "")
                _txt(tr, "Valor", f"{t.valor:.2f}" if t.valor is not None else "")

        if n.observacoes:
            _txt(doc, "Observacoes", n.observacoes)

    ET.indent(root, space="  ")
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)

    return StreamingResponse(
        io.BytesIO(xml_bytes),
        media_type="application/xml",
        headers={"Content-Disposition": "attachment; filename=notas_fiscais.xml"},
    )


@router.get("/", response_model=List[NotaFiscalOut])
async def list_notas(skip: int = 0, limit: int = 50,
                     tipo: Optional[str] = None,
                     data_inicio: Optional[datetime] = None,
                     data_fim: Optional[datetime] = None,
                     db: Session = Depends(get_db),
                     _: User = Depends(get_current_user)):
    q = db.query(NotaFiscal).filter(NotaFiscal.confirmado == True)  # noqa: E712
    if tipo:
        q = q.filter(NotaFiscal.tipo == tipo)
    if data_inicio:
        q = q.filter(NotaFiscal.data_importacao >= data_inicio)
    if data_fim:
        q = q.filter(NotaFiscal.data_importacao <= data_fim)
    return q.order_by(NotaFiscal.data_importacao.desc()).offset(skip).limit(limit).all()


@router.get("/{nota_id}", response_model=NotaFiscalOut)
async def get_nota(nota_id: int, db: Session = Depends(get_db),
                   _: User = Depends(get_current_user)):
    nota = db.query(NotaFiscal).filter(NotaFiscal.id == nota_id).first()
    if not nota:
        raise HTTPException(status_code=404, detail="Nota não encontrada")
    return nota


@router.put("/{nota_id}", response_model=NotaFiscalOut)
async def update_nota(nota_id: int, data: NotaFiscalCreate,
                      db: Session = Depends(get_db),
                      _: User = Depends(get_current_user)):
    nota = db.query(NotaFiscal).filter(NotaFiscal.id == nota_id).first()
    if not nota:
        raise HTTPException(status_code=404, detail="Nota não encontrada")
    for k, v in data.model_dump(exclude_unset=True).items():
        setattr(nota, k, v)
    db.commit()
    db.refresh(nota)
    return nota


@router.delete("/{nota_id}")
async def delete_nota(nota_id: int, db: Session = Depends(get_db),
                      _: User = Depends(get_current_user)):
    nota = db.query(NotaFiscal).filter(NotaFiscal.id == nota_id).first()
    if not nota:
        raise HTTPException(status_code=404, detail="Nota não encontrada")
    db.delete(nota)
    db.commit()
    return {"ok": True}
