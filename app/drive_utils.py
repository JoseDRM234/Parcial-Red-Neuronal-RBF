# ==================== drive_utils.py ====================
"""
Helpers para Google Drive (pydrive2) con autenticación directa.
"""
import os

try:
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive
    Pydrive_available = True
except Exception:
    Pydrive_available = False

def init_drive():
    """Inicializa y autentica con Google Drive usando OAuth2"""
    if not Pydrive_available:
        raise RuntimeError("pydrive2 no está instalado. Instale con: pip install pydrive2")
    
    gauth = GoogleAuth()
    # Autenticación local con navegador web
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)
    return drive

def list_drive_files(query=None):
    """Lista archivos en Drive. query ejemplo: "title contains 'dataset'"""
    drive = init_drive()
    file_list = drive.ListFile({'q': query or "trashed=false"}).GetList()
    return [(f['title'], f['id']) for f in file_list]

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
