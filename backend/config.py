import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite_default")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 480))
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "tesseract")
TESSDATA_PREFIX = os.getenv("TESSDATA_PREFIX", "")
SEFAZ_TIMEOUT = 5  # seconds
OCR_CONFIDENCE_THRESHOLD = 60
