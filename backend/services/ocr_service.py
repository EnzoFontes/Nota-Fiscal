import logging
import os
import pytesseract
import cv2
import numpy as np
from PIL import Image
from config import TESSERACT_CMD, TESSDATA_PREFIX, OCR_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

try:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
except Exception:
    pass

if TESSDATA_PREFIX:
    os.environ['TESSDATA_PREFIX'] = TESSDATA_PREFIX

# PSM 4 → single column (good for bank/PIX receipts with label:value pairs)
# PSM 6 → uniform block of text (good for NF-e/NFC-e printed receipts)
_TESS_CONFIG_COLUMN = '--psm 4 --oem 3'
_TESS_CONFIG_BLOCK = '--psm 6 --oem 3'


def _get_ocr_lang() -> str:
    """Return 'por+eng' if Portuguese tessdata is available, else 'eng'."""
    try:
        langs = pytesseract.get_languages(config='')
        if 'por' in langs:
            return 'por+eng'
    except Exception:
        pass
    logger.warning(
        'Tesseract Portuguese (por) not found — using English. '
        'Download por.traineddata for better accuracy.'
    )
    return 'eng'


_OCR_LANG = _get_ocr_lang()


def _is_screenshot(img: np.ndarray) -> bool:
    """Heuristic: screenshots have very uniform, near-white background (>65% pixels >240)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    bright_ratio = np.sum(gray > 240) / gray.size
    return bright_ratio > 0.65


def _crop_receipt(img: np.ndarray) -> np.ndarray:
    """
    Isolate the white receipt from a dark/textured background (e.g. wooden table).

    Uses Gaussian blur to suppress wood-grain texture, then Otsu auto-threshold
    which finds the optimal split point regardless of lighting conditions.
    A large morphological close fills crumple/fold gaps inside the receipt mask.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    # Blur suppresses the wood-grain texture that confuses thresholding
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # CLAHE boosts local contrast so paper stands out from background
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(blurred)

    # Otsu automatically picks the threshold — no hardcoded value needed
    _, mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Large close kernel fills crumple holes and folds inside the receipt region
    close_size = max(25, img.shape[1] // 60)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    total = img.shape[0] * img.shape[1]

    if area < total * 0.15:
        logger.warning('Crop: receipt region < 15%% of image, using full image')
        return img

    x, y, w, h = cv2.boundingRect(largest)
    pad = max(25, img.shape[1] // 80)
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(img.shape[1] - x, w + 2 * pad)
    h = min(img.shape[0] - y, h + 2 * pad)
    cropped = img[y:y+h, x:x+w]
    logger.info(f'Crop: {img.shape[:2]} → {cropped.shape[:2]} (receipt {area/total*100:.0f}%% of original)')
    return cropped


def _sharpen(gray: np.ndarray) -> np.ndarray:
    """Apply unsharp mask for sharpening blurry images."""
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    return cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)


def _deskew(gray: np.ndarray) -> np.ndarray:
    """
    Rotate image to correct small tilt angles (≤ 20°).

    Uses Otsu binarization to reliably isolate dark text pixels from light
    paper regardless of raw gray values, then computes the skew angle from
    the minimum-area bounding rectangle of all text pixels.
    """
    # Otsu finds text pixels robustly without a hardcoded threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 200:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.3 or abs(angle) > 20:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Adaptive preprocessing for OCR.

    Three paths:
    1. Screenshot (app comprovante): mild sharpen + upscale.
    2. Physical photo of receipt (NFC-e, NF-e, printed cupom):
       crop → sharpen → CLAHE → adaptive threshold → cleanup.
    3. PDF-rendered page: same as screenshot (clean white background).
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((100, 100), dtype=np.uint8)

    # Cap input size to avoid OOM on large phone photos (>3000px side)
    _MAX_INPUT = 3000
    h0, w0 = img.shape[:2]
    if max(h0, w0) > _MAX_INPUT:
        scale = _MAX_INPUT / max(h0, w0)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    if _is_screenshot(img):
        # ── Screenshot / PDF-render path ─────────────────────────────────────
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        h, w = sharpened.shape
        if w < 1200:
            scale = 1200 / w
            sharpened = cv2.resize(sharpened, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_CUBIC)
        return sharpened

    # ── Physical photo path ───────────────────────────────────────────────────
    # Step 1: Isolate receipt from background (fixed Otsu-based crop)
    img = _crop_receipt(img)

    # Step 2: Upscale if image is small
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side < 2000:
        scale = 2000 / long_side
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 3: Deskew using Otsu-detected text pixels (not raw gray values)
    gray = _deskew(gray)

    # Step 4: Sharpen
    gray = _sharpen(gray)

    # Step 5: CLAHE — adaptive histogram equalization for uneven lighting / shadows
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Step 6: Denoise (h=7 preserves more fine print detail than h=10)
    denoised = cv2.fastNlMeansDenoising(gray, h=7)

    # Step 7: Adaptive threshold
    # blockSize is scaled to ~1/50 of image height so it remains meaningful
    # on large phone-photo images; forced to an odd number, minimum 11.
    h_img = denoised.shape[0]
    block_size = max(11, (h_img // 50) | 1)  # | 1 ensures odd number
    logger.info(f'Adaptive threshold blockSize={block_size} (image h={h_img})')

    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=block_size, C=9
    )

    # Step 8: Light morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary


def _run_ocr(pil_img: Image.Image, config: str) -> tuple[str, float]:
    """Run Tesseract once, returning (text, avg_confidence) from a single call."""
    data = pytesseract.image_to_data(
        pil_img, lang=_OCR_LANG, config=config,
        output_type=pytesseract.Output.DICT,
    )
    confidences = [int(c) for c in data['conf'] if int(c) > 0]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0

    # Reconstruct text with line breaks from block/line metadata
    lines: dict = {}
    for word, conf, block, par, line in zip(
        data['text'], data['conf'],
        data['block_num'], data['par_num'], data['line_num'],
    ):
        if str(word).strip() and int(conf) > 0:
            lines.setdefault((block, par, line), []).append(word)
    text = '\n'.join(' '.join(words) for words in lines.values())

    return text, round(avg_conf, 1)


def extract_text_from_image(image_bytes: bytes) -> dict:
    """
    Run Tesseract OCR with two PSM modes; return the result with higher confidence.

    PSM 4 (single column) suits PIX/bank screenshots.
    PSM 6 (block of text) suits printed NFC-e / NF-e receipts.
    """
    try:
        processed = preprocess_image(image_bytes)
        pil_img = Image.fromarray(processed)

        text4, conf4 = _run_ocr(pil_img, _TESS_CONFIG_COLUMN)
        text6, conf6 = _run_ocr(pil_img, _TESS_CONFIG_BLOCK)

        if conf6 > conf4 + 2:
            text, avg_confidence = text6, conf6
            logger.info(f'OCR psm=6 chosen (conf6={conf6} > conf4={conf4})')
        else:
            text, avg_confidence = text4, conf4
            logger.info(f'OCR psm=4 chosen (conf4={conf4} >= conf6={conf6})')

        logger.info(f'OCR lang={_OCR_LANG} confidence={avg_confidence:.1f}%% chars={len(text)}')
        logger.debug(f'OCR text:\n{text}')

        return {
            'text': text,
            'confidence': avg_confidence,
            'accepted': avg_confidence >= OCR_CONFIDENCE_THRESHOLD,
        }
    except Exception as e:
        logger.error(f'OCR error: {e}')
        return {'text': '', 'confidence': 0.0, 'accepted': False}
