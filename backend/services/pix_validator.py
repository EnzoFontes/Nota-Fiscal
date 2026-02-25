"""
PIX document validation service.

Inspired by ValidadorPixDocsAI (github.com/ronieremarques/ValidadorPixDocsAI).
Non-AI validation rules ported to Python. Claude API fraud analysis added
as an optional enhancement using the user's Anthropic subscription.
"""

import re
import os
import io
import logging
import struct
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Keyword rules (extended from ValidadorPixDocsAI) ──────────────────────────
_PIX_KEYWORDS = [
    'pix', 'txid', 'comprovante', 'transferência', 'transferencia',
    'pagamento', 'valor', 'pago', 'recebido', 'banco',
    'chave pix', 'id da transação', 'id da transacao',
    'nu pagamentos', 'nubank', 'tipo de transferência',
]

_REQUIRED_KEYWORD_COUNT = 2  # at least 2 must appear


def check_keywords(text: str) -> Tuple[bool, List[str]]:
    """Return (passed, found_keywords)."""
    t = text.lower()
    found = [k for k in _PIX_KEYWORDS if k in t]
    return len(found) >= _REQUIRED_KEYWORD_COUNT, found


# ── Text integrity checks ─────────────────────────────────────────────────────

def check_text_integrity(text: str) -> List[str]:
    """Return list of warning strings (empty = clean)."""
    warnings: List[str] = []

    # Irregular spacing — sign of copy-paste manipulation
    if re.search(r'\s{3,}', text):
        warnings.append('Espaçamento irregular detectado no texto')

    # Non-standard characters (outside standard PT/BR charset)
    unusual = re.findall(
        r'[^\w\s\.,\-\/\(\)\$\%\:\;\!\?\"\'\@\#\&\*\+\=\[\]\{\}\|áéíóúâêîôûãõçàèìòùäëïöüÁÉÍÓÚÂÊÎÔÛÃÕÇÀÈÌÒÙÄËÏÖÜ]',
        text
    )
    if unusual:
        warnings.append(f'Caracteres não usuais detectados: {set(unusual)}')

    return warnings


# ── Monetary value extraction ─────────────────────────────────────────────────

def extract_all_valores(text: str) -> List[float]:
    """Extract all monetary amounts from OCR text (Brazilian format)."""
    matches = re.findall(r'R?\$?\s*\d+[.,]\d{2}', text)
    values = []
    for m in matches:
        clean = re.sub(r'[R$\s]', '', m).replace(',', '.')
        try:
            values.append(float(clean))
        except ValueError:
            pass
    return values


# ── Date extraction ───────────────────────────────────────────────────────────

def extract_all_dates(text: str) -> List[str]:
    """Extract all date strings from OCR text."""
    return re.findall(r'\d{2}[\/\-]\d{2}[\/\-](?:\d{4}|\d{2})', text)


# ── Image metadata checks (PNG/JPEG) ─────────────────────────────────────────

def _read_png_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
    """Read width/height from PNG header without Pillow."""
    if data[:8] != b'\x89PNG\r\n\x1a\n':
        return None
    try:
        w = struct.unpack('>I', data[16:20])[0]
        h = struct.unpack('>I', data[20:24])[0]
        return w, h
    except Exception:
        return None


def _read_jpeg_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
    """Read width/height from JPEG SOF marker."""
    if data[:2] != b'\xff\xd8':
        return None
    i = 2
    while i < len(data) - 8:
        if data[i] != 0xff:
            break
        marker = data[i + 1]
        length = struct.unpack('>H', data[i + 2:i + 4])[0]
        if marker in (0xC0, 0xC2):  # SOF0, SOF2
            h = struct.unpack('>H', data[i + 5:i + 7])[0]
            w = struct.unpack('>H', data[i + 7:i + 9])[0]
            return w, h
        i += 2 + length
    return None


_COMMON_RATIOS = [16 / 9, 16 / 10, 4 / 3, 21 / 9, 19 / 9]
_COMMON_RESOLUTIONS = [
    (1920, 1080), (2560, 1440), (3840, 2160),
    (1366, 768), (1280, 720), (1440, 900),
    (2880, 1800), (1792, 828), (2340, 1080),
]


def check_image_metadata(image_bytes: bytes) -> Dict:
    """
    Check image dimensions for common screenshot resolutions.
    Returns a dict with: is_screenshot (bool), width, height, warnings list.
    """
    result = {
        'is_screenshot': False,
        'width': None,
        'height': None,
        'warnings': [],
    }

    dims = _read_png_dimensions(image_bytes) or _read_jpeg_dimensions(image_bytes)
    if not dims:
        result['warnings'].append('Não foi possível ler as dimensões da imagem')
        return result

    w, h = dims
    result['width'] = w
    result['height'] = h

    # Too small to be a real screenshot
    if w < 300 or h < 300:
        result['warnings'].append('Resolução muito baixa para uma captura de tela')

    # File too small
    if len(image_bytes) < 50 * 1024:
        result['warnings'].append('Arquivo muito pequeno (possível imagem editada)')

    # Check common screen aspect ratios (±5%)
    if h > 0:
        ratio = w / h
        for common in _COMMON_RATIOS:
            if abs(ratio - common) / common < 0.05:
                result['is_screenshot'] = True
                break

    # Check exact known resolutions (±10%)
    if not result['is_screenshot']:
        for cw, ch in _COMMON_RESOLUTIONS:
            if abs(w - cw) / cw < 0.1 and abs(h - ch) / ch < 0.1:
                result['is_screenshot'] = True
                break

    return result


# ── Full validation pipeline ──────────────────────────────────────────────────

def validate_pix_document(
    text: str,
    image_bytes: Optional[bytes] = None,
) -> Dict:
    """
    Run all non-AI validation rules. Returns a structured report.

    Fields:
      is_valid  — bool
      score     — 0–100 (higher = more trustworthy)
      keywords  — list of found PIX keywords
      valores   — list of monetary values found
      datas     — list of dates found
      warnings  — list of warning strings
    """
    warnings: List[str] = []

    # 1. Keywords
    kw_passed, found_kw = check_keywords(text)
    if not kw_passed:
        warnings.append(
            f'Poucos termos PIX encontrados ({len(found_kw)}): {found_kw}'
        )

    # 2. Monetary values
    valores = extract_all_valores(text)
    if not valores:
        warnings.append('Nenhum valor monetário encontrado no texto')

    # 3. Dates
    datas = extract_all_dates(text)
    if not datas:
        warnings.append('Nenhuma data encontrada no texto')

    # 4. Text integrity
    text_warnings = check_text_integrity(text)
    warnings.extend(text_warnings)

    # 5. Image metadata (optional)
    meta = {}
    if image_bytes:
        meta = check_image_metadata(image_bytes)
        warnings.extend(meta.get('warnings', []))

    # Scoring: start at 100, deduct per warning
    deductions = {
        'Poucos termos': 30,
        'Nenhum valor': 20,
        'Nenhuma data': 15,
        'Espaçamento': 10,
        'Caracteres': 10,
        'Resolução': 5,
        'Arquivo muito pequeno': 5,
    }
    score = 100
    for w in warnings:
        for key, pts in deductions.items():
            if key in w:
                score -= pts
                break
    score = max(0, score)

    # A document is valid if it has keywords + a value (dates optional for PIX)
    is_valid = kw_passed and len(valores) > 0

    return {
        'is_valid': is_valid,
        'score': score,
        'keywords': found_kw,
        'valores': valores,
        'datas': datas,
        'warnings': warnings,
        'metadata': meta,
    }


# ── Claude API fraud analysis ─────────────────────────────────────────────────

def analyze_with_claude(text: str, validation_report: Dict) -> Optional[str]:
    """
    Optional: send OCR text + rule-based report to Claude claude-haiku-4-5 for
    fraud/authenticity analysis. Returns the model's assessment string,
    or None if the API key is not configured.

    Set ANTHROPIC_API_KEY in the environment to enable this feature.
    """
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        return None

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        prompt = f"""Você é um especialista em fraudes de comprovantes PIX brasileiros.
Analise o texto extraído via OCR de um comprovante e o relatório de validação automática abaixo.

=== TEXTO OCR ===
{text[:2000]}

=== RELATÓRIO DE VALIDAÇÃO ===
Keywords encontradas: {validation_report.get('keywords')}
Valores monetários: {validation_report.get('valores')}
Datas: {validation_report.get('datas')}
Alertas: {validation_report.get('warnings')}

Responda objetivamente:
1. O comprovante parece autêntico ou manipulado?
2. Há inconsistências no texto (valores, datas, nomes, chave PIX)?
3. Conclusão final: AUTÊNTICO, SUSPEITO ou INVÁLIDO — com justificativa breve."""

        message = client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=400,
            messages=[{'role': 'user', 'content': prompt}],
        )
        return message.content[0].text

    except ImportError:
        logger.warning('anthropic package not installed. Run: pip install anthropic')
        return None
    except Exception as e:
        logger.error(f'Claude API error: {e}')
        return None
