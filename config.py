import os
import json
import tempfile
import platform

# Constants
LOCATION = "us-central1"
MODEL_NAME = "gemini-2.5-pro"
EMBEDDING_MODEL_NAME = "text-embedding-004"
UNIVERSE_DOMAIN = "googleapis.com"

def setup_tesseract():
    """Configures Tesseract path on Windows."""
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

def setup_gcp_credentials(secrets=None):
    """
    Configures Google Cloud credentials.
    Prioritizes GCP_CREDENTIALS env var, then service_account env vars, then streamlit secrets.
    Returns the PROJECT_ID if successful, raises Exception otherwise.
    """
    credentials_configured = False
    project_id = None
    credentials_dict = None

    # 1. Try GCP_CREDENTIALS env var (JSON string)
    if os.getenv('GCP_CREDENTIALS'):
        try:
            credentials_json = os.getenv('GCP_CREDENTIALS')
            credentials_dict = json.loads(credentials_json)
            project_id = credentials_dict.get('project_id')
        except Exception as e:
            raise Exception(f"Failed to load GCP credentials from environment: {e}")

    # 2. Try individual env vars
    elif os.getenv('type') == 'service_account':
        try:
            credentials_dict = {
                'type': os.getenv('type'),
                'project_id': os.getenv('project_id'),
                'private_key_id': os.getenv('private_key_id'),
                'private_key': os.getenv('private_key'),
                'client_email': os.getenv('client_email'),
                'client_id': os.getenv('client_id'),
                'auth_uri': os.getenv('auth_uri'),
                'token_uri': os.getenv('token_uri'),
                'auth_provider_x509_cert_url': os.getenv('auth_provider_x509_cert_url'),
                'client_x509_cert_url': os.getenv('client_x509_cert_url'),
                'universe_domain': os.getenv('universe_domain', UNIVERSE_DOMAIN)
            }
            project_id = os.getenv('project_id')
        except Exception as e:
            raise Exception(f"Failed to load credentials from individual environment variables: {e}")

    # 3. Try secrets (passed from Streamlit)
    elif secrets and 'gcp_service_account' in secrets:
        try:
            credentials_dict = dict(secrets['gcp_service_account'])
            project_id = credentials_dict.get('project_id')
        except Exception as e:
            raise Exception(f"Failed to load Streamlit secrets: {e}")

    if credentials_dict:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(credentials_dict, f)
                credentials_path = f.name
            
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            return project_id
        except Exception as e:
             raise Exception(f"Failed to write credentials file: {e}")
    
    raise Exception("Google Cloud credentials not found.")
