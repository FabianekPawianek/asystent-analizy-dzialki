import os
import platform

LOCATION = "us-central1"
MODEL_NAME = "gemini-3.1-flash-lite-preview"
EMBEDDING_MODEL_NAME = "text-embedding-004"
UNIVERSE_DOMAIN = "googleapis.com"

def setup_tesseract():
    if platform.system() == 'Windows':
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', ''))
        ]
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    import pytesseract
                    pytesseract.pytesseract.tesseract_cmd = path
                    return True
                except ImportError:
                    pass
    return False

def get_google_api_key(secrets=None):
    api_key = os.getenv('GOOGLE_API_KEY')

    if not api_key and secrets:
        try:
            api_key = secrets.get('GOOGLE_API_KEY')
        except Exception:
            api_key = None

    if not api_key:
        env_candidates = [
            os.path.join(os.getcwd(), '.env'),
            os.path.join(os.path.dirname(__file__), '.env'),
        ]
        for path in env_candidates:
            try:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith('#'):
                                continue
                            if line.startswith('GOOGLE_API_KEY'):
                                value = line.split('=', 1)[1].strip()
                                if value.startswith(('"', "'")) and value.endswith(('"', "'")):
                                    value = value[1:-1]
                                api_key = value
                                os.environ['GOOGLE_API_KEY'] = api_key
                                break
            except Exception:
                continue
            if api_key:
                break

    if not api_key:
        raise Exception("Brak GOOGLE_API_KEY. Dodaj do .env lub do Streamlit Secrets.")

    return api_key
