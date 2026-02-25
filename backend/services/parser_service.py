import re
from datetime import datetime
from typing import Optional, Dict, List

# Portuguese month abbreviations used in bank receipts (e.g. "25 FEV 2026")
_MONTHS_PT = {
    'JAN': 1, 'FEV': 2, 'MAR': 3, 'ABR': 4, 'MAI': 5, 'JUN': 6,
    'JUL': 7, 'AGO': 8, 'SET': 9, 'OUT': 10, 'NOV': 11, 'DEZ': 12,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _br_float(s: str) -> Optional[float]:
    """Convert Brazilian number string to float. Handles '1.234,56' and '89,99'."""
    if not s:
        return None
    s = s.strip().lstrip('R$').strip()
    # Remove thousands separator, replace decimal comma with dot
    if ',' in s:
        s = s.replace('.', '').replace(',', '.')
    try:
        return float(s)
    except ValueError:
        return None


def _extract(text: str, pattern: str, flags: int = 0) -> Optional[str]:
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None


def _extract_all(text: str, pattern: str, flags: int = 0) -> List[str]:
    return [m.group(1).strip() for m in re.finditer(pattern, text, flags)]


# ── CNPJ / CPF ───────────────────────────────────────────────────────────────

def parse_cnpj(text: str) -> Optional[str]:
    m = re.search(r'\d{2}[\.\s]?\d{3}[\.\s]?\d{3}[\/\s]?\d{4}[-\s]?\d{2}', text)
    return re.sub(r'[^\d]', '', m.group()) if m else None


def parse_cpf(text: str) -> Optional[str]:
    """Generic CPF — 11 digits with optional separators."""
    m = re.search(r'\d{3}[\.\s]?\d{3}[\.\s]?\d{3}[-\s]?\d{2}(?!\d)', text)
    return re.sub(r'[^\d]', '', m.group()) if m else None


def parse_cpf_consumidor(text: str) -> Optional[str]:
    """
    CPF from NFC-e footer line: 'CONSUMIDOR - CPF 708.898.801-15'
    Also handles: 'CPF do Consumidor: ...'
    """
    m = re.search(
        r'(?:CONSUMIDOR\s*[-–]\s*CPF|CPF\s+do\s+Consumidor)\s*:?\s*([\d.\-]+)',
        text, re.IGNORECASE,
    )
    if m:
        return re.sub(r'[^\d]', '', m.group(1))
    return None


# ── Monetary values ───────────────────────────────────────────────────────────

def parse_valor(text: str) -> Optional[float]:
    """Generic: first monetary value found in text."""
    # Standard: R$ 1.053,65
    m = re.search(r'R\$\s*([\d\.]+,\d{2})', text)
    if m:
        return _br_float(m.group(1))
    # Without R$: 1.053,65 or 89,99
    m = re.search(r'(?<!\d)([\d]{1,3}(?:\.\d{3})*,\d{2})(?!\d)', text)
    if m:
        return _br_float(m.group(1))
    # OCR artifact: R$ 105365 (no separators, 5-7 digits → implicit 2 decimals)
    m = re.search(r'R\$\s*(\d{5,7})(?!\d)', text)
    if m:
        raw = m.group(1)
        return float(raw[:-2] + '.' + raw[-2:])
    return None


def parse_valor_pagar(text: str) -> Optional[float]:
    """
    Extract 'Valor a Pagar' or 'VALOR PAGO' amount specifically.
    More reliable than generic parse_valor for NFC-e receipts.
    """
    for pattern in [
        r'Valor\s+a\s+Pagar\s+R\$\s*([\d.,]+)',
        r'Valor\s+a\s+Pagar[:\s]+([\d.,]+)',
        r'VALOR\s+PAGO\s+([\d.,]+)',
        r'TOTAL\s+R\$\s*([\d.,]+)',
        r'TOTAL\s+PAGAR[:\s]+([\d.,]+)',
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            v = _br_float(m.group(1))
            if v is not None:
                return v
    return None


def parse_desconto(text: str) -> Optional[float]:
    """Extract total discount amount from NF/NFC-e."""
    # "DESCONTO ... R$ -40,00" or "R$ -40.00"
    m = re.search(r'DESCONTO\b.*?R\$\s*-?([\d.,]+)', text, re.IGNORECASE | re.DOTALL)
    if m:
        return _br_float(m.group(1))
    # "Desconto por incentivo no item    40,00"
    m = re.search(r'Desconto\s+por\s+incentivo.*?([\d.,]+)\s*$', text,
                  re.IGNORECASE | re.MULTILINE)
    if m:
        return _br_float(m.group(1))
    return None


# ── Document identifiers ──────────────────────────────────────────────────────

def parse_chave_acesso(text: str) -> Optional[str]:
    """44-digit access key (chave de acesso NF-e/NFC-e)."""
    clean = re.sub(r'\s', '', text)
    m = re.search(r'\d{44}', clean)
    return m.group() if m else None


def parse_numero_nf(text: str) -> Optional[str]:
    """
    Extract NF-e/NFC-e number from multiple formats:
      - 'NFC-e n. 000017454'
      - 'NF-e n.º 000017454'
      - 'Número: 17454'
      - 'DOC 20568'   (NFC-e cupom number on the header line)
    """
    for pattern in [
        r'NF[Ce]{0,2}[-\s]*[eE]\s+n\.?[º°]?\s*(\d+)',
        r'N[úu]mero\s*:?\s*(\d+)',
        r'\bDOC\s*[=:]?\s*(\d{4,})\b',
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).lstrip('0') or m.group(1)
    return None


def parse_serie_nf(text: str) -> Optional[str]:
    """Extract NF-e/NFC-e series: 'Serie 113' or 'Série: 003'."""
    m = re.search(r'S[eé]r[ié]e?\s*:?\s*(\d{1,3})\b', text, re.IGNORECASE)
    return m.group(1) if m else None


def parse_razao_social(text: str) -> Optional[str]:
    """
    Extract emitente razão social from top of receipt.
    Looks for an all-caps or title-case business name in the first 10 lines,
    skipping address/CNPJ/phone lines.
    """
    skip_patterns = re.compile(
        r'CNPJ|CPF|FONE|TEL[EF]|FAX|CEP|RUA|AV\.|QUADRA|BLOCO|SETOR|'
        r'LOTE|SALA|Documento\s+Auxiliar|Nota\s+Fiscal|PROCON|'
        r'\d{2}[./]\d{3}[./]\d{3}|\d{5}-\d{3}',
        re.IGNORECASE,
    )
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    for line in lines[:12]:
        if (len(line) >= 5
                and not skip_patterns.search(line)
                and re.search(r'[A-ZÁÉÍÓÚÂÊÎÔÛÃÕÇ]', line)):
            # Accept if it looks like a name: letters, spaces, S/A, Ltda, &, /
            if re.match(r'^[A-Za-zÀ-ÖØ-öø-ÿ0-9\s&/.\'-]+$', line):
                return line
    return None


# ── Date ─────────────────────────────────────────────────────────────────────

def parse_date(text: str) -> Optional[datetime]:
    """Try multiple date formats commonly found in Brazilian documents."""
    # Portuguese month abbreviations: "25 FEV 2026"
    m = re.search(
        r'(\d{1,2})\s+(JAN|FEV|MAR|ABR|MAI|JUN|JUL|AGO|SET|OUT|NOV|DEZ)\s+(\d{4})',
        text, re.IGNORECASE,
    )
    if m:
        try:
            return datetime(int(m.group(3)), _MONTHS_PT[m.group(2).upper()], int(m.group(1)))
        except ValueError:
            pass

    for pattern, fmt in [
        (r'(\d{2}/\d{2}/\d{4})', '%d/%m/%Y'),
        (r'(\d{2}-\d{2}-\d{4})', '%d-%m-%Y'),
        (r'(\d{4}-\d{2}-\d{2})', '%Y-%m-%d'),
        (r'(\d{2}/\d{2}/\d{2})\b', '%d/%m/%y'),   # short year: "24/12/25"
    ]:
        m = re.search(pattern, text)
        if m:
            try:
                return datetime.strptime(m.group(1), fmt)
            except ValueError:
                continue
    return None


# ── PIX / NFC-e classification ────────────────────────────────────────────────

def is_pix_receipt(text: str) -> bool:
    t = text.lower()
    keywords = [
        'pix',
        'txid',
        'chave pix',
        'comprovante de transfer',
        'id da transa',
        'tipo de transfer',
        'nu pagamentos',
        'transferência pix',
    ]
    return any(k in t for k in keywords)


def is_nfce(text: str) -> bool:
    """Detect NFC-e (Nota Fiscal de Consumidor Eletrônica)."""
    t = text.lower()
    return (
        'nota fiscal de consumidor' in t
        or 'nfc-e' in t
        or 'cupom fiscal' in t
        or 'consumidor eletronica' in t
        or 'nfce' in t
    )


# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_pix_receipt(text: str) -> Dict:
    recebedor_nome = (
        _extract_after_section(text, 'Destino', 'Nome')
        or _extract(text, r'Recebedor[:\s]+(.+?)(?:\n|CPF|CNPJ)', re.IGNORECASE)
    )
    pagador_nome = (
        _extract_after_section(text, 'Origem', 'Nome')
        or _extract(text, r'Pagador[:\s]+(.+?)(?:\n|CPF|CNPJ)', re.IGNORECASE)
    )
    chave = _extract(text, r'Chave\s*Pix[:\s]+([^\n]+)', re.IGNORECASE)
    if chave:
        chave = chave.strip()

    txid = _extract(
        text,
        r'(?:TXID|ID\s+da\s+transa[çc][ãa]o)\s*[:\n]+\s*([A-Za-z0-9]{10,})',
        re.IGNORECASE,
    )

    return {
        'tipo': 'pix',
        'valor_total': parse_valor(text),
        'data_emissao': parse_date(text),
        'pix_txid': txid,
        'pix_pagador_nome': pagador_nome,
        'pix_pagador_cpf_cnpj': parse_cpf(text) or parse_cnpj(text),
        'pix_recebedor_nome': recebedor_nome,
        'pix_recebedor_chave': chave,
    }


def parse_nf_image(text: str) -> Dict:
    """
    Parse NF-e or NFC-e from OCR text of a physical receipt photo.

    Covers the receipt format in the reference image (Dona de Casa S/A NFC-e):
      - Emitente CNPJ and razão social at top
      - 'Valor a Pagar R$ 89,99' as the total
      - Chave de Acesso in spaced 44-digit blocks
      - 'CONSUMIDOR - CPF ...'
      - 'NFC-e n. 000017454 Serie 113'
      - Date in short format '24/12/25 17:49:22'
      - 'DESCONTO' line for discounts
    """
    # Prefer 'Valor a Pagar' over the first R$ value found
    valor = parse_valor_pagar(text) or parse_valor(text)
    desconto = parse_desconto(text)
    numero = parse_numero_nf(text)
    serie = parse_serie_nf(text)

    # CPF do consumidor (NFC-e footer)
    destinatario_cpf = parse_cpf_consumidor(text) or parse_cpf(text)

    return {
        'numero': numero,
        'serie': serie,
        'chave_acesso': parse_chave_acesso(text),
        'data_emissao': parse_date(text),
        'emitente_cnpj': parse_cnpj(text),
        'emitente_razao_social': parse_razao_social(text),
        'destinatario_cpf_cnpj': destinatario_cpf,
        'valor_total': valor,
        'valor_desconto': desconto,
    }


# ── Section-based extraction helper (for PIX bank receipts) ──────────────────

# Section boundary markers used when parsing bank receipt layouts (e.g. Nubank).
_SECTION_STOPPERS = [
    'Destino', 'Origem', 'Nu Pagamentos',
    'ID da transaç', 'ID da transac', 'Ouvidoria', 'CNPJ',
]


def _extract_after_section(text: str, section: str, field: str) -> Optional[str]:
    """
    Extract a field value from within a named section of a bank receipt.
    Nubank format example:
        Destino
        Nome   ENZO BATISTA FONTES
        CPF    •••.105.301-••
    """
    sec_m = re.search(rf'\b{re.escape(section)}\b', text, re.IGNORECASE)
    if not sec_m:
        return None

    slice_start = sec_m.end()
    section_text = text[slice_start:]

    stop_pos = len(section_text)
    for stopper in _SECTION_STOPPERS:
        if stopper.lower() == section.lower():
            continue
        sm = re.search(rf'\b{re.escape(stopper)}\b', section_text, re.IGNORECASE)
        if sm and sm.start() < stop_pos:
            stop_pos = sm.start()

    section_text = section_text[:stop_pos]

    fm = re.search(rf'\b{re.escape(field)}\b\s+([^\n]+)', section_text, re.IGNORECASE)
    if fm:
        return fm.group(1).strip()

    fm = re.search(rf'\b{re.escape(field)}\b\s*\n\s*([^\n]+)', section_text, re.IGNORECASE)
    if fm:
        return fm.group(1).strip()

    return None
