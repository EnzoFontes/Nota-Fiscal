"""
PDF processing service using PyMuPDF (fitz).

Two strategies:
  1. Text PDF (gerado digitalmente): extract text directly — instant, 100% accurate.
  2. Scanned PDF (foto ou escaneado): render each page to image, then run OCR pipeline.
"""

import logging
import io
from typing import Optional
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# DPI for rendering scanned PDF pages → images for OCR
_RENDER_DPI = 200


def _page_has_useful_text(page: fitz.Page, min_chars: int = 50) -> bool:
    """Return True if the page contains enough selectable text."""
    text = page.get_text("text")
    return len(text.strip()) >= min_chars


def extract_text_from_pdf(pdf_bytes: bytes) -> Optional[tuple[str, int]]:
    """
    Extract text from a digital (non-scanned) PDF.

    Returns (combined_text, page_count), or None if the PDF has no selectable
    text (i.e. it is a scanned PDF and OCR should be used instead).
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_text = []
        has_text = False

        for page in doc:
            if _page_has_useful_text(page):
                has_text = True
            pages_text.append(page.get_text("text"))

        page_count = doc.page_count
        doc.close()

        if not has_text:
            return None

        combined = "\n".join(pages_text)
        logger.info(f"PDF text extraction: {len(combined)} chars from {page_count} page(s)")
        return combined, page_count

    except Exception as e:
        logger.error(f"PDF text extraction error: {e}")
        return None


def render_pdf_pages(pdf_bytes: bytes) -> list[bytes]:
    """
    Render each PDF page to a PNG image (bytes).
    Used for scanned/image PDFs that have no selectable text.
    """
    images: list[bytes] = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        matrix = fitz.Matrix(_RENDER_DPI / 72, _RENDER_DPI / 72)  # 72 dpi is PDF default

        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY)
            png_bytes = pix.tobytes("png")
            images.append(png_bytes)
            logger.info(f"PDF page {i+1} rendered: {pix.width}x{pix.height}px")

        doc.close()
    except Exception as e:
        logger.error(f"PDF render error: {e}")

    return images


def process_pdf(pdf_bytes: bytes) -> dict:
    """
    Process a PDF and return the best available text.

    Returns:
      {
        'text': str,          — extracted / OCR text (all pages combined)
        'is_scanned': bool,   — True if OCR was needed
        'page_count': int,
      }
    """
    # Try direct text extraction first
    extraction = extract_text_from_pdf(pdf_bytes)
    if extraction is not None:
        text, page_count = extraction
        return {'text': text, 'is_scanned': False, 'page_count': page_count}

    # Scanned PDF — render pages and OCR them
    logger.info("PDF has no selectable text — will OCR rendered pages")
    page_images = render_pdf_pages(pdf_bytes)

    if not page_images:
        return {'text': '', 'is_scanned': True, 'page_count': 0}

    # Lazy import to avoid circular dependency
    from services.ocr_service import extract_text_from_image

    all_text_parts = []
    best_confidence = 0.0

    for i, img_bytes in enumerate(page_images):
        result = extract_text_from_image(img_bytes)
        if result['text'].strip():
            all_text_parts.append(result['text'])
        best_confidence = max(best_confidence, result['confidence'])
        logger.info(f"Page {i+1} OCR: confidence={result['confidence']:.1f}%")

    return {
        'text': '\n\n'.join(all_text_parts),
        'is_scanned': True,
        'page_count': len(page_images),
        'confidence': best_confidence,
    }
