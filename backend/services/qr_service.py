"""
QR code reading and PIX BR Code (EMV) parsing service.

Supports:
  - NF-e / NFC-e QR codes (URL with chave de acesso de 44 dígitos)
  - PIX BR Code estático e dinâmico (formato EMV ISO 18004)
"""

import re
import logging
import numpy as np
import cv2
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


# ── QR Code detection ─────────────────────────────────────────────────────────

def _decode_with_opencv(img: np.ndarray) -> List[str]:
    """Try OpenCV's built-in QR detector (fast, no extra deps)."""
    detector = cv2.QRCodeDetector()
    # detectAndDecodeMulti returns (retval, decoded_list, points, straight_qrcodes)
    try:
        ok, decoded_list, _, _ = detector.detectAndDecodeMulti(img)
        if ok and decoded_list:
            return [d for d in decoded_list if d]
    except Exception as e:
        logger.debug(f"OpenCV QR multi failed: {e}")
    # Fallback: single QR
    try:
        data, _, _ = detector.detectAndDecode(img)
        if data:
            return [data]
    except Exception as e:
        logger.debug(f"OpenCV QR single failed: {e}")
    return []


def _decode_with_pyzbar(img: np.ndarray) -> List[str]:
    """Try pyzbar — more robust for angled/low-quality photos."""
    try:
        from pyzbar import pyzbar
        from PIL import Image
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img)
        results = pyzbar.decode(pil)
        return [r.data.decode('utf-8', errors='replace') for r in results if r.data]
    except ImportError:
        return []
    except Exception as e:
        logger.debug(f"pyzbar failed: {e}")
        return []


def read_qr_codes(image_bytes: bytes) -> List[str]:
    """
    Decode all QR codes in an image. Returns list of decoded strings.
    Tries multiple scales and preprocessing to improve detection rate.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    results: List[str] = []

    # Try original image first
    results.extend(_decode_with_opencv(img))
    results.extend(_decode_with_pyzbar(img))

    if not results:
        # Try grayscale + binary threshold (helps with printed QRs)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.extend(_decode_with_opencv(binary))
        results.extend(_decode_with_pyzbar(binary))

    if not results:
        # Try upscaled version (helps when QR is small in the image)
        h, w = img.shape[:2]
        scale = max(1.0, 1500 / max(h, w))
        if scale > 1.1:
            upscaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            results.extend(_decode_with_opencv(upscaled))
            results.extend(_decode_with_pyzbar(upscaled))

    # Deduplicate preserving order
    seen = set()
    unique = []
    for r in results:
        if r and r not in seen:
            seen.add(r)
            unique.append(r)

    if unique:
        logger.info(f"QR codes found: {len(unique)}")
        for r in unique:
            logger.debug(f"QR content: {r[:120]}")
    else:
        logger.info("No QR codes detected in image")

    return unique


# ── NF-e / NFC-e QR Code parser ───────────────────────────────────────────────

def parse_nfe_qr(payload: str) -> Optional[Dict]:
    """
    Extract chave de acesso from NF-e / NFC-e QR code URL.

    Examples:
      https://www.sefaz.rs.gov.br/NFCE/NFCE-COM.aspx?chave=43210...&nVerificador=...
      https://nfce.fazenda.mg.gov.br/portalnfce/sistema/qrcode.xhtml?p=31210...|2|1|1|HASH
      https://www.nfce.fazenda.sp.gov.br/NFCeConsultaPublica/Paginas/ConsultaQRCode.aspx?chave=35210...
    """
    if not ('sefaz' in payload.lower() or 'nfce' in payload.lower() or 'nfe' in payload.lower()):
        return None

    # chave= parameter
    m = re.search(r'chave=(\d{44})', payload, re.IGNORECASE)
    if m:
        return {'tipo': 'nfe', 'chave_acesso': m.group(1), 'qr_url': payload}

    # p= parameter: "CHAVE44DIGITS|2|1|1|HASH"
    m = re.search(r'[?&]p=(\d{44})', payload, re.IGNORECASE)
    if m:
        return {'tipo': 'nfe', 'chave_acesso': m.group(1), 'qr_url': payload}

    # Bare 44-digit number anywhere in a URL payload
    clean = re.sub(r'\s', '', payload)
    m = re.search(r'\d{44}', clean)
    if m and ('http' in payload.lower() or 'www' in payload.lower()):
        return {'tipo': 'nfe', 'chave_acesso': m.group(), 'qr_url': payload}

    return None


# ── PIX BR Code (EMV) parser ──────────────────────────────────────────────────

def _parse_emv(payload: str) -> Dict[str, str]:
    """
    Parse EMV TLV structure: each field is ID(2) + LEN(2) + VALUE(N).
    Returns flat dict {id: value} including nested MAI fields.
    """
    fields: Dict[str, str] = {}
    i = 0
    while i + 4 <= len(payload):
        tag = payload[i:i+2]
        try:
            length = int(payload[i+2:i+4])
        except ValueError:
            break
        value = payload[i+4:i+4+length]
        fields[tag] = value
        i += 4 + length
    return fields


def parse_pix_qr(payload: str) -> Optional[Dict]:
    """
    Parse a PIX BR Code (static or dynamic) from its raw string payload.

    Returns dict with extracted PIX fields, or None if not a valid PIX QR.

    BR Code field IDs:
      00 - Payload Format Indicator (must be "01")
      01 - Point of Initiation Method (11=static, 12=dynamic)
      26-51 - Merchant Account Information (MAI) for br.gov.bcb.pix
        00 - GUI
        01 - PIX key (chave) — static QR
        02 - Description
        25 - URL — dynamic QR
      52 - Merchant Category Code
      53 - Transaction Currency
      54 - Transaction Amount
      58 - Country Code
      59 - Merchant Name
      60 - Merchant City
      61 - Postal Code
      62 - Additional Data (nested)
        05 - Reference Label (TXID)
      63 - CRC16
    """
    # Sanity check: must start with "000201" (Payload Format Indicator = 01)
    if not payload.startswith('000201'):
        return None

    top = _parse_emv(payload)

    # Find MAI block for br.gov.bcb.pix (tags 26–51)
    mai: Dict[str, str] = {}
    for tag_id in [f'{i:02d}' for i in range(26, 52)]:
        if tag_id in top:
            sub = _parse_emv(top[tag_id])
            # Check GUI
            if 'br.gov.bcb.pix' in sub.get('00', '').lower():
                mai = sub
                break

    if not mai and '26' not in top:
        # Not a PIX QR
        return None

    # Static QR: field 01 = PIX key
    # Dynamic QR: field 25 = URL for fetching payment details
    pix_key = mai.get('01', '').strip()
    pix_url = mai.get('25', '').strip()   # dynamic
    description = mai.get('02', '').strip()

    # Transaction amount (field 54)
    valor: Optional[float] = None
    if '54' in top:
        try:
            valor = float(top['54'])
        except ValueError:
            pass

    # Merchant name (field 59) = recebedor
    merchant_name = top.get('59', '').strip()

    # TXID from Additional Data field 62, sub-field 05
    txid: Optional[str] = None
    if '62' in top:
        add = _parse_emv(top['62'])
        txid = add.get('05', '').strip() or None

    # Method: 11=static, 12=dynamic
    method = top.get('01', '')
    is_dynamic = method == '12'

    return {
        'tipo': 'pix',
        'pix_recebedor_nome': merchant_name or None,
        'pix_recebedor_chave': pix_key or pix_url or None,
        'valor_total': valor,
        'pix_txid': txid,
        'pix_qr_descricao': description or None,
        'pix_qr_dinamico': is_dynamic,
        'pix_qr_url': pix_url or None,
    }


# ── Unified entry point ───────────────────────────────────────────────────────

def extract_from_qr(image_bytes: bytes) -> Optional[Dict]:
    """
    Read QR codes from an image and return parsed fields (NF-e or PIX).
    Returns None if no QR code is found or none is recognized.
    """
    payloads = read_qr_codes(image_bytes)
    for payload in payloads:
        # Try PIX first (more specific format check)
        result = parse_pix_qr(payload)
        if result:
            logger.info(f"PIX QR parsed: chave={result.get('pix_recebedor_chave')} valor={result.get('valor_total')}")
            return result

        # Try NF-e
        result = parse_nfe_qr(payload)
        if result:
            logger.info(f"NF-e QR parsed: chave_acesso={result.get('chave_acesso')}")
            return result

    return None
