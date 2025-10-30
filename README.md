RBF App - Tkinter
====================================

Requisitos
----------
- Python 3.13
- pip install -r requirements.txt

Descripción
-----------
Aplicación de escritorio para entrenar y simular una Red RBF:
- Carga dataset (local / Drive)
- Definir número inicial de centros (Depende de las entradas del dataset)
- Añadir centros manualmente (o random)
- Entrena (una sola iteración, mínimos cuadrados)
- Muestra: Centros finales, Pesos finales, Matriz A, EG / MAE / RMSE
- Guarda resultados en CSV/JSON y modelo .pkl (para cargar y sólo simular)
- Subida/descarga a Google Drive (opcional, requiere client_secrets.json + PyDrive2)

Ejecución
---------
1. Instala dependencias:
   pip install -r requirements.txt
2. Ejecuta:
   python main.py

