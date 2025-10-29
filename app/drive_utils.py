# ==================== drive_utils.py ====================
"""
Utilidades completas para conexión con Google Drive usando pydrive2.
Incluye autenticación, listado, carga directa (sin descargar) y subida de archivos.
"""

import os
import io
import pandas as pd

try:
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive
    Pydrive_available = True
except Exception:
    Pydrive_available = False

_drive = None  # Cache global de sesión Drive


# ============================================================
# 🔐 AUTENTICACIÓN
# ============================================================
def init_drive():
    """Inicializa y autentica con Google Drive usando OAuth2"""
    global _drive

    if not Pydrive_available:
        raise RuntimeError("❌ pydrive2 no está instalado. Instale con: pip install pydrive2")

    if _drive is not None:
        return _drive

    gauth = GoogleAuth()

    # Intentar usar credenciales guardadas
    if os.path.exists("credentials.json"):
        gauth.LoadCredentialsFile("credentials.json")

    if gauth.credentials is None:
        gauth.LocalWebserverAuth()  # Se abre en navegador
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()

    gauth.SaveCredentialsFile("credentials.json")
    _drive = GoogleDrive(gauth)
    return _drive


# ============================================================
# 📂 LISTAR Y SUBIR ARCHIVOS
# ============================================================
def list_drive_files(query=None):
    """Lista archivos en Drive. query ejemplo: "title contains 'dataset'"""
    drive = init_drive()
    file_list = drive.ListFile({'q': query or "trashed=false"}).GetList()
    return [(f['title'], f['id'], f['mimeType']) for f in file_list]


def upload_to_drive(local_path, folder_id=None):
    """Sube archivo a Drive y retorna el file_id"""
    drive = init_drive()
    file_name = os.path.basename(local_path)
    metadata = {'title': file_name}
    if folder_id:
        metadata['parents'] = [{'id': folder_id}]

    gfile = drive.CreateFile(metadata)
    gfile.SetContentFile(local_path)
    gfile.Upload()
    return f"✅ Subido a Drive: {file_name} (ID: {gfile['id']})"


def download_from_drive(file_id, save_path):
    """Descarga archivo desde Drive usando su ID"""
    drive = init_drive()
    gfile = drive.CreateFile({'id': file_id})
    gfile.GetContentFile(save_path)
    return save_path


# ============================================================
# ⚡ NUEVO: LECTURA DIRECTA DESDE DRIVE (SIN DESCARGAR)
# ============================================================
def read_file_from_drive(file_id):
    """
    Lee un archivo directamente desde Google Drive en memoria (sin guardarlo localmente).
    Devuelve un DataFrame si es CSV/XLSX/JSON o el contenido de texto si es otro tipo.
    """
    drive = init_drive()
    gfile = drive.CreateFile({'id': file_id})
    name = gfile['title'].lower()

    # --- CSV ---
    if name.endswith(".csv"):
        content = gfile.GetContentString()
        return pd.read_csv(io.StringIO(content))

    # --- XLSX ---
    elif name.endswith(".xlsx"):
        content = gfile.GetContentBinary()
        return pd.read_excel(io.BytesIO(content))

    # --- JSON ---
    elif name.endswith(".json"):
        content = gfile.GetContentString()
        try:
            # Si es un modelo JSON entrenado o un dataset JSON
            import json
            data = json.loads(content)
            # Si tiene estructura de modelo
            if "centers" in data and "weights" in data:
                return data
            else:
                return pd.read_json(io.StringIO(content))
        except Exception:
            return pd.read_json(io.StringIO(content))

    # --- Texto plano ---
    else:
        return gfile.GetContentString()
