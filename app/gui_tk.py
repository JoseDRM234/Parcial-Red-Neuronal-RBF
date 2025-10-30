
"""
Interfaz gráfica completa con todas las funcionalidades del PDF
"""
import os
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# IMPORTS RELATIVOS
from .model_rbf import RBFModel
from .data_manager import (
    read_dataset, detect_xy_from_df, preprocess,
    split_train_test, save_results, save_model_json_csv, load_model_json,
    RESULTS_DIR, MODELS_DIR, DATASETS_DIR
)
from .drive_utils import upload_to_drive, read_file_from_drive, list_drive_files, download_from_drive, Pydrive_available
import json

class RBFApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sistema RBF - Examen Práctico IA")
        self.geometry("1200x800")
        self.configure(bg="#f2f4f6")

        # Estado de la aplicación
        self.X = None
        self.Y = None
        self.meta = None
        self.scaler = None
        self.model = None
        self.last_eval = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

        # Variables de configuración
        self.n_centers_var = tk.IntVar(value=3)
        self.train_pct_var = tk.DoubleVar(value=70.0)
        self.error_opt_var = tk.DoubleVar(value=0.1)

        self._build_ui()

    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except:
            pass
        
        style.configure("TButton", font=("Segoe UI", 10), padding=6)
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TEntry", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 12, "bold"))

        # Notebook (pestañas)
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=10)

        # Crear tabs
        self.tab_load = ttk.Frame(nb)
        self.tab_config = ttk.Frame(nb)
        self.tab_train = ttk.Frame(nb)
        self.tab_sim = ttk.Frame(nb)
        self.tab_graph = ttk.Frame(nb)
        self.tab_results = ttk.Frame(nb)

        nb.add(self.tab_load, text="📂 1. Cargar Dataset")
        nb.add(self.tab_config, text="⚙️ 2. Configuración")
        nb.add(self.tab_train, text="🧠 3. Entrenamiento")
        nb.add(self.tab_sim, text="🔎 4. Simulación")
        nb.add(self.tab_graph, text="📈 5. Gráficas")
        nb.add(self.tab_results, text="📊 6. Resultados")

        self._build_tab_load()
        self._build_tab_config()
        self._build_tab_train()
        self._build_tab_sim()
        self._build_tab_graph()
        self._build_tab_results()

    # ==================== TAB 1: CARGA ====================
    def _build_tab_load(self):
        f = self.tab_load
        
        # Título
        ttk.Label(f, text="Carga de Dataset", style="Title.TLabel").pack(pady=10)
        
        # Frame para carga local
        frm_local = ttk.LabelFrame(f, text="📁 Carga Local", padding=10)
        frm_local.pack(fill="x", padx=20, pady=10)
        
        ttk.Button(frm_local, text="Seleccionar archivo CSV/XLSX/JSON", 
                  command=self.load_local_dataset, width=30).pack(pady=5)
        
        # Frame para Google Drive
        frm_drive = ttk.LabelFrame(f, text="☁️ Google Drive", padding=10)
        frm_drive.pack(fill="x", padx=20, pady=10)
        
        ttk.Button(frm_drive, text="🔐 Autenticar y listar archivos", 
                  command=self.authenticate_drive, width=30).pack(pady=5)
        
        ttk.Label(frm_drive, text="Archivos disponibles:").pack(anchor="w", pady=5)
        self.drive_listbox = tk.Listbox(frm_drive, height=5)
        self.drive_listbox.pack(fill="x", pady=5)
        
           # Botón para cargar directamente
        ttk.Button(frm_drive, text="📥 Cargar seleccionado",
                command=self.load_from_drive_direct, width=30).pack(pady=5)
        
        ttk.Button(frm_drive, text="📥 Descargar seleccionado", 
                  command=self.download_selected_from_drive).pack(pady=5)
        
        # Información del dataset
        ttk.Label(f, text="📋 Información del Dataset:").pack(anchor="w", padx=20, pady=(10,5))
        self.load_info = tk.Text(f, height=10, width=100, font=("Consolas", 9))
        self.load_info.pack(padx=20, pady=5)

    def load_local_dataset(self):
    # Abrir diálogo de selección de archivo
        path = filedialog.askopenfilename(
            filetypes=[
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx"),
                ("JSON files", "*.json")
            ]
        )

        if not path:
            return  # Si no selecciona nada, salir

        try:

            # Detectar tipo de archivo y cargar
            ext = os.path.splitext(path)[1].lower()
            if ext == ".csv":
                df = pd.read_csv(path)
            elif ext in [".xls", ".xlsx"]:
                df = pd.read_excel(path)
            elif ext == ".json":
                df = pd.read_json(path)
            else:
                raise ValueError("Formato de archivo no soportado")

            # Procesamiento de datos
            self.X, self.Y, self.meta = detect_xy_from_df(df)
            self.X, self.scaler = preprocess(self.X)

            # Mostrar información del dataset en la interfaz
            self.load_info.delete("1.0", tk.END)
            self.load_info.insert(tk.END, f"✅ Dataset cargado: {os.path.basename(path)}\n\n")
            self.load_info.insert(tk.END, f"📊 Estadísticas:\n")
            self.load_info.insert(tk.END, f"  - Patrones totales: {self.meta['n_patterns']}\n")
            self.load_info.insert(tk.END, f"  - Número de entradas: {self.meta['n_inputs']}\n")
            self.load_info.insert(tk.END, f"  - Número de salidas: {self.meta['n_outputs']}\n")
            self.load_info.insert(tk.END, f"  - Columnas X: {self.meta['X_cols']}\n")
            self.load_info.insert(tk.END, f"  - Columna Y: {self.meta['Y_col']}\n\n")
            self.load_info.insert(tk.END, "✅ Preprocesamiento: Normalización Z-score aplicada\n")

            # Reiniciar modelo y evaluación anterior
            self.model = None
            self.last_eval = None

            messagebox.showinfo("Éxito", "Dataset cargado correctamente")

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Error al cargar dataset:\n{str(e)}")
            
    # ==================== Dentro de la clase RBFApp ====================

    def authenticate_drive(self):
        """Autentica con Google Drive y lista archivos compatibles (CSV, XLSX, JSON, modelo)"""
        if not Pydrive_available:
            messagebox.showerror("Error", "pydrive2 no está instalado.\nInstale con: pip install pydrive2")
            return
        try:
             # 🔹 Filtra solo archivos .csv o .json en Google Drive
            query = (
                "trashed=false and "
                "(title contains '.csv' or title contains '.json')"
            )
            files = list_drive_files(query)
            self.drive_files = files
            # Limpiar lista
            self.drive_listbox.delete(0, tk.END)

            for title, fid, mime in files:
                # Mostrar el tipo MIME al usuario (útil para diferenciar datasets o modelos)
                if "csv" in mime or "excel" in mime:
                    tag = "📊 Dataset"
                elif "json" in mime:
                    tag = "🧠 Modelo/JSON"
                else:
                    tag = "📄 Otro"
                self.drive_listbox.insert(tk.END, f"{tag}  {title}  ({fid[:10]}...)")

            messagebox.showinfo("Drive", f"✅ Autenticado. {len(files)} archivos encontrados.")

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error Drive", f"❌ Error durante autenticación:\n{str(e)}")

    def show_matrix_popup(self, A, cols_header):
        """Muestra la matriz A en una ventana emergente tipo tabla"""
        import tkinter as tk
        from tkinter import ttk

        win = tk.Toplevel(self)
        win.title("📋 Matriz A - Vista Completa")
        win.geometry("1000x600")

        ttk.Label(win, text="📋 Matriz A (activaciones radiales)", font=("Segoe UI", 11, "bold")).pack(pady=8)

        # Frame principal con scroll
        frame = ttk.Frame(win)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Scrollbars
        scroll_y = ttk.Scrollbar(frame, orient="vertical")
        scroll_y.pack(side="right", fill="y")
        scroll_x = ttk.Scrollbar(frame, orient="horizontal")
        scroll_x.pack(side="bottom", fill="x")

        # Treeview (tabla)
        tree = ttk.Treeview(
            frame,
            columns=cols_header,
            show="headings",
            yscrollcommand=scroll_y.set,
            xscrollcommand=scroll_x.set,
            height=25
        )

        # Configurar encabezados
        for col in cols_header:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor="center")

        # Insertar filas
        for i, row in enumerate(A):
            formatted = [f"{v:.6f}" for v in row]
            tree.insert("", "end", values=formatted)

        # Enlazar scrolls
        tree.pack(fill="both", expand=True)
        scroll_y.config(command=tree.yview)
        scroll_x.config(command=tree.xview)

        ttk.Button(win, text="Cerrar", command=win.destroy).pack(pady=5)

# ==================== Dentro de la clase RBFApp ====================
    def load_from_drive_direct(self):
        sel = self.drive_listbox.curselection()
        if not sel:
            messagebox.showwarning("Aviso", "Seleccione un archivo de la lista")
            return

        idx = sel[0]
        title, fid, mime = self.drive_files[idx]

        try:
            df_or_json = read_file_from_drive(fid)

            # 🔹 Si es un modelo (contiene pesos y centros)
            if isinstance(df_or_json, dict) and "weights" in df_or_json:
                from .model_rbf import RBFModel
                import numpy as np
                from sklearn.preprocessing import StandardScaler

                model = RBFModel(n_centers=df_or_json.get("n_centers", 2))
                model.centers = np.array(df_or_json.get("centers", []))
                model.weights = np.array(df_or_json.get("weights", []))
                model.trained = True

                scaler = None
                if df_or_json.get("scaler_mean") and df_or_json.get("scaler_std"):
                    scaler = StandardScaler()
                    scaler.mean_ = np.array(df_or_json["scaler_mean"])
                    scaler.scale_ = np.array(df_or_json["scaler_std"])
                    scaler.n_features_in_ = len(scaler.mean_)

                self.model = model
                self.scaler = scaler
                self.meta = df_or_json.get("meta")
                self.last_eval = df_or_json.get("last_eval")

                messagebox.showinfo("Modelo cargado", f"✅ Modelo '{title}' abierto desde Drive")

            else:
                # 🔹 Dataset (CSV, Excel o JSON)
                df = df_or_json
                self.X, self.Y, self.meta = detect_xy_from_df(df)
                self.X, self.scaler = preprocess(self.X)

                # 🔸 Mostrar la misma información que el modo local
                self.load_info.delete("1.0", tk.END)
                self.load_info.insert(tk.END, f"✅ Dataset cargado desde Drive: {title}\n\n")
                self.load_info.insert(tk.END, f"📊 Estadísticas:\n")
                self.load_info.insert(tk.END, f"  - Patrones totales: {self.meta['n_patterns']}\n")
                self.load_info.insert(tk.END, f"  - Número de entradas: {self.meta['n_inputs']}\n")
                self.load_info.insert(tk.END, f"  - Número de salidas: {self.meta['n_outputs']}\n")
                self.load_info.insert(tk.END, f"  - Columnas X: {self.meta['X_cols']}\n")
                self.load_info.insert(tk.END, f"  - Columna Y: {self.meta['Y_col']}\n\n")
                self.load_info.insert(tk.END, "✅ Preprocesamiento: Normalización Z-score aplicada\n")

                # 🔸 Reiniciar modelo y resultados anteriores
                self.model = None
                self.last_eval = None

                messagebox.showinfo("Éxito", f"✅ Dataset '{title}' cargado correctamente desde Drive")

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"❌ Error al leer archivo desde Drive:\n{str(e)}")

    def download_selected_from_drive(self):
        sel = self.drive_listbox.curselection()
        if not sel:
            messagebox.showwarning("Aviso", "Seleccione un archivo de la lista")
            return
        
        try:
            idx = sel[0]
            title, fid = self.drive_files[idx]
            
            tmp_path = os.path.join(DATASETS_DIR, title)
            download_from_drive(fid, tmp_path)
            
            # Cargar el dataset descargado
            df = read_dataset(tmp_path)
            self.X, self.Y, self.meta = detect_xy_from_df(df)
            self.X, self.scaler = preprocess(self.X)
            
            self.load_info.delete("1.0", tk.END)
            self.load_info.insert(tk.END, f"✅ Dataset descargado desde Drive: {title}\n\n")
            self.load_info.insert(tk.END, f"📊 Estadísticas:\n")
            self.load_info.insert(tk.END, f"  - Patrones: {self.meta['n_patterns']}\n")
            self.load_info.insert(tk.END, f"  - Entradas: {self.meta['n_inputs']}\n")
            self.load_info.insert(tk.END, f"  - Salidas: {self.meta['n_outputs']}\n")
            
            messagebox.showinfo("Éxito", "Dataset descargado y cargado")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ==================== TAB 2: CONFIGURACIÓN ====================
    def _build_tab_config(self):
        f = self.tab_config
        
        ttk.Label(f, text="Configuración de la Red RBF", style="Title.TLabel").pack(pady=10)
        
        # Frame de configuración
        frm_cfg = ttk.LabelFrame(f, text="⚙️ Parámetros", padding=15)
        frm_cfg.pack(fill="x", padx=20, pady=10)
        
        # Número de centros
        ttk.Label(frm_cfg, text="Número de centros radiales:").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Spinbox(frm_cfg, from_=2, to=20, textvariable=self.n_centers_var, width=10).grid(row=0, column=1, sticky="w", padx=10)
        
        # Porcentaje entrenamiento
        ttk.Label(frm_cfg, text="Porcentaje de entrenamiento (%):").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Spinbox(frm_cfg, from_=50, to=90, textvariable=self.train_pct_var, width=10).grid(row=1, column=1, sticky="w", padx=10)
        
        # Error óptimo
        ttk.Label(frm_cfg, text="Error óptimo (EG):").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(frm_cfg, textvariable=self.error_opt_var, width=10).grid(row=2, column=1, sticky="w", padx=10)
        
        # Gestión de centros
        frm_centers = ttk.LabelFrame(f, text="🎯 Gestión de Centros", padding=15)
        frm_centers.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Tabla de centros
        cols = ("idx", "coords")
        self.centers_tree = ttk.Treeview(frm_centers, columns=cols, show="headings", height=8)
        self.centers_tree.heading("idx", text="Centro #")
        self.centers_tree.heading("coords", text="Coordenadas")
        self.centers_tree.column("idx", width=80, anchor="center")
        self.centers_tree.column("coords", width=700, anchor="w")
        self.centers_tree.pack(side="left", fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(frm_centers, orient="vertical", command=self.centers_tree.yview)
        self.centers_tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        
        # Botones de gestión
        frm_btns = ttk.Frame(f)
        frm_btns.pack(fill="x", padx=20, pady=10)
        
        ttk.Button(frm_btns, text="🎲 Inicializar aleatorios", 
                  command=self.init_centers_random).pack(side="left", padx=5)
        ttk.Button(frm_btns, text="➕ Agregar aleatorio", 
                  command=self.add_center_random).pack(side="left", padx=5)
        ttk.Button(frm_btns, text="✏️ Agregar manual", 
                  command=self.add_center_manual).pack(side="left", padx=5)
        ttk.Button(frm_btns, text="🗑️ Eliminar seleccionado", 
                  command=self.remove_selected_center).pack(side="left", padx=5)

    def _clear_centers_view(self):
        for item in self.centers_tree.get_children():
            self.centers_tree.delete(item)

    def _populate_centers_view(self, centers):
        self._clear_centers_view()
        if centers is None:
            return
        for i, c in enumerate(centers, start=1):
            coords_str = ", ".join([f"{float(x):.6f}" for x in c])
            self.centers_tree.insert("", "end", values=(f"Centro {i}", coords_str))

    def init_centers_random(self):
        """Inicializa los centros radiales aleatoriamente con validación profesional."""
        if self.X is None:
            messagebox.showwarning("Aviso", "Cargue un dataset primero.")
            return

        try:
            n = int(self.n_centers_var.get())
            n_inputs = self.X.shape[1]

            # Validar mínimo 2 centros
            if n < 2:
                messagebox.showwarning(
                    "Aviso",
                    "Debe tener al menos 2 centros radiales.\n"
                    "El valor se ajustará automáticamente a 2."
                )
                n = 2
                self.n_centers_var.set(2)

            # Validar que no sea menor que el número de entradas
            elif n < n_inputs:
                messagebox.showwarning(
                    "Aviso",
                    f"El número de centros ({n}) es menor que el número de variables de entrada ({n_inputs}).\n"
                    f"Se ajustará automáticamente a {n_inputs}."
                )
                n = n_inputs
                self.n_centers_var.set(n_inputs)

            # Inicializar el modelo
            self.model = RBFModel(n_centers=n, random_state=42)
            self.model.initialize_centers(self.X)

            # Mostrar centros en la vista
            self._populate_centers_view(self.model.centers)

            messagebox.showinfo(
                "Centros inicializados",
                f"✅ {n} centros radiales inicializados correctamente."
            )

        except ValueError:
            messagebox.showerror("Error", "Ingrese un número válido para los centros radiales.")
        except Exception as e:
            messagebox.showerror("Error", f"Error al inicializar centros:\n{str(e)}")


    def add_center_random(self):
        """Agrega un nuevo centro radial aleatorio dentro del rango de los datos."""
        if self.X is None:
            messagebox.showwarning("Aviso", "Cargue un dataset primero.")
            return

        try:
            if self.model is None:
                self.model = RBFModel(n_centers=0, random_state=42)
                self.model.centers = np.empty((0, self.X.shape[1]))

            mins = self.X.min(axis=0)
            maxs = self.X.max(axis=0)
            coords = np.random.uniform(mins, maxs)
            self.model.add_center(coords)

            self._populate_centers_view(self.model.centers)

            messagebox.showinfo(
                "Centro agregado",
                f"✅ Centro radial aleatorio agregado.\n\n"
                f"Total actual: {self.model.n_centers} centros."
            )

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo agregar el centro:\n{str(e)}")


    def add_center_manual(self):
        """Permite al usuario ingresar manualmente un nuevo centro radial."""
        if self.X is None:
            messagebox.showwarning("Aviso", "Cargue un dataset primero.")
            return

        n_features = self.X.shape[1]
        prompt = f"Ingrese {n_features} valores separados por comas:"
        s = simpledialog.askstring("Centro Manual", prompt)

        if not s:
            return

        try:
            parts = [float(x.strip()) for x in s.split(",")]
            if len(parts) != n_features:
                messagebox.showerror(
                    "Error",
                    f"Debe ingresar exactamente {n_features} valores numéricos."
                )
                return

            if self.model is None:
                self.model = RBFModel(n_centers=0, random_state=42)
                self.model.centers = np.empty((0, n_features))

            self.model.add_center(parts)
            self._populate_centers_view(self.model.centers)

            messagebox.showinfo(
                "Centro agregado",
                f"✅ Centro manual agregado correctamente.\n\n"
                f"Total actual: {self.model.n_centers} centros."
            )

        except ValueError:
            messagebox.showerror("Error", "Debe ingresar solo valores numéricos separados por comas.")
        except Exception as e:
            messagebox.showerror("Error", f"Error al agregar centro manual:\n{str(e)}")


    def remove_selected_center(self):
        sel = self.centers_tree.selection()
        if not sel:
            messagebox.showwarning("Aviso", "Seleccione un centro")
            return
        
        if self.model is None or self.model.centers is None:
            return
        
        idx = self.centers_tree.index(sel[0])
        centers_list = list(self.model.centers)
        centers_list.pop(idx)
        
        if centers_list:
            self.model.centers = np.vstack(centers_list)
            self.model.n_centers = len(centers_list)
        else:
            self.model.centers = None
            self.model.n_centers = 0
        
        self._populate_centers_view(self.model.centers)
        messagebox.showinfo("Centro", "✅ Centro eliminado")

    # ==================== TAB 3: ENTRENAMIENTO ====================
    def _build_tab_train(self):
        f = self.tab_train

        ttk.Label(f, text="Entrenamiento de la Red RBF", style="Title.TLabel").pack(pady=10)

        # ------------------------------------------------------------
        # Botones principales
        # ------------------------------------------------------------
        frm_btns = ttk.Frame(f)
        frm_btns.pack(fill="x", padx=20, pady=10)

        ttk.Button(
            frm_btns,
            text="🚀 Entrenar Modelo",
            command=self.train_model,
            width=20
        ).pack(side="left", padx=5)

        ttk.Button(
            frm_btns,
            text="💾 Guardar Modelo (JSON/CSV)",
            command=self.save_model_dialog,
            width=25
        ).pack(side="left", padx=5)

        ttk.Button(
            frm_btns,
            text="☁️ Subir a Drive",
            command=self.upload_model_to_drive,
            width=15
        ).pack(side="left", padx=5)
         # Botón que mostrará la matriz (inicialmente deshabilitado)
        self.btn_show_matrix = ttk.Button(f, text="📊 Ver Matriz A Completa", state="disabled",
                                        command=self._show_matrix_popup_event)
        self.btn_show_matrix.pack(pady=5)

        # ------------------------------------------------------------
        # Área de resultados (Log de entrenamiento)
        # ------------------------------------------------------------
        ttk.Label(f, text="📋 Log de Entrenamiento:").pack(anchor="w", padx=20, pady=(10, 5))
        self.txt_train = tk.Text(f, height=28, width=120, font=("Consolas", 9))
        self.txt_train.pack(padx=20, pady=5, fill="both", expand=True)

    def _show_matrix_popup_event(self):
        """Evento al presionar el botón de ver matriz completa"""
        if hasattr(self, "A") and self.A is not None:
            self.show_matrix_popup(self.A, self.cols_header)
        else:
            messagebox.showwarning("Aviso", "Primero entrene el modelo para generar la matriz A.")

    def train_model(self):
        if self.X is None:
            messagebox.showerror("Error", "Cargue un dataset primero")
            return

        try:
            # ============================================================
            # PASO 1: Validación de configuración
            # ============================================================
            n_centers = int(self.n_centers_var.get())
            n_inputs = self.X.shape[1]

            if n_centers < n_inputs:
                messagebox.showwarning(
                    "Configuración inválida",
                    f"⚠️ El número de centros ({n_centers}) debe ser al menos igual "
                    f"al número de entradas del dataset ({n_inputs}).\n\n"
                    f"💡 Recomendación: Use al menos {n_inputs} centros."
                )
                return

            # ============================================================
            # PASO 2: Preparación general
            # ============================================================
            train_pct = self.train_pct_var.get() / 100
            test_pct = 1 - train_pct

            self.txt_train.delete("1.0", tk.END)
            self.txt_train.insert(tk.END, "=" * 80 + "\n")
            self.txt_train.insert(tk.END, "ENTRENAMIENTO DE RED NEURONAL RBF\n")
            self.txt_train.insert(tk.END, "=" * 80 + "\n\n")

            # ============================================================
            # PASO 3: Partición del dataset
            # ============================================================
            self.txt_train.insert(tk.END, "📊 PASO 3: Partición del Dataset\n")
            self.X_train, self.Y_train, self.X_test, self.Y_test, split_info = split_train_test(
                self.X, self.Y, train_pct=train_pct, random_state=42
            )
            self.txt_train.insert(tk.END, f"  ✅ Entrenamiento: {split_info['N_train']} patrones ({split_info['pct_train']:.1f}%)\n")
            self.txt_train.insert(tk.END, f"  ✅ Prueba: {split_info['N_test']} patrones ({split_info['pct_test']:.1f}%)\n\n")

            # ============================================================
            # PASO 4: Configuración de la red
            # ============================================================
            self.txt_train.insert(tk.END, "⚙️ PASO 4: Configuración de la Red\n")
            if self.model is None or self.model.centers is None:
                self.model = RBFModel(n_centers=n_centers, random_state=42, use_pdf=True)
                self.model.initialize_centers(self.X_train)
                self.txt_train.insert(tk.END, f"  ✅ Centros inicializados: {n_centers}\n")
            else:
                self.txt_train.insert(tk.END, f"  ✅ Usando centros configurados: {self.model.n_centers}\n")

            self.txt_train.insert(tk.END, f"  ✅ Función de activación: FA(d) = d² * ln(d)\n")
            self.txt_train.insert(tk.END, f"  ✅ Error óptimo: {self.error_opt_var.get()}\n\n")

            # ============================================================
            # PASO 5-6: Cálculo de Distancias y Activaciones
            # ============================================================
            self.txt_train.insert(tk.END, "🔢 PASO 5-6: Cálculo de Distancias y Activaciones\n")

            # Entrenamiento del modelo (obtiene matriz A y pesos)
            A, weights = self.model.train(self.X_train, self.Y_train)

            # Crear carpeta de resultados si no existe
            os.makedirs(RESULTS_DIR, exist_ok=True)

            # Mostrar información general de la matriz A
            self.txt_train.insert(tk.END, f"  ✅ Matriz A construida correctamente\n")
            self.txt_train.insert(tk.END, f"     Dimensiones: {A.shape[0]} patrones x {A.shape[1]} columnas\n")
            self.txt_train.insert(tk.END, f"     Columnas → [1 (bias) + {self.model.n_centers} activaciones radiales]\n\n")

            # Construir encabezados de columnas
            cols_header = ["Bias"] + [f"Φ{j+1}" for j in range(self.model.n_centers)]

            # Mostrar primeras 10 filas en el log (resumen)
            self.txt_train.insert(tk.END, "📋 MATRIZ A (primeras 10 filas):\n")
            self.txt_train.insert(tk.END, "-" * 100 + "\n")
            self.txt_train.insert(tk.END, " | ".join([f"{h:^12}" for h in cols_header]) + "\n")
            self.txt_train.insert(tk.END, "-" * 100 + "\n")

            for i, row in enumerate(A[:10]):  # Solo 10 filas para no saturar
                formatted_row = " | ".join([f"{val:>12.6f}" for val in row])
                self.txt_train.insert(tk.END, f"Fila {i+1:>3}: {formatted_row}\n")

            if A.shape[0] > 10:
                self.txt_train.insert(tk.END, f"... ({A.shape[0]-10} filas adicionales ocultas)\n")

            self.txt_train.insert(tk.END, "-" * 100 + "\n\n")

            # Guardar matriz A en CSV
            import pandas as pd
            df_A = pd.DataFrame(A, columns=cols_header)
            path_A = os.path.join(RESULTS_DIR, "matriz_A.csv")
            df_A.to_csv(path_A, index=False)
            self.txt_train.insert(tk.END, f"💾 Matriz A guardada en: {path_A}\n\n")
        

            # ============================================================
            # PASO 6: Cálculo de Pesos
            # ============================================================
            self.txt_train.insert(tk.END, "🧠 PASO 7: Cálculo de Pesos (Mínimos Cuadrados)\n")
            self.txt_train.insert(tk.END, f"  W₀ (bias) = {weights[0]:.8f}\n")
            for i, w in enumerate(weights[1:], 1):
                self.txt_train.insert(tk.END, f"  W{i} = {w:.8f}\n")
            self.txt_train.insert(tk.END, "\n")

            # ============================================================
            # PASO 7: Evaluación del Modelo
            # ============================================================
            self.txt_train.insert(tk.END, "📊 PASO 9: Evaluación del Modelo\n\n")
            eval_train = self.model.evaluate(self.X_train, self.Y_train)
            eval_test = self.model.evaluate(self.X_test, self.Y_test)

            # ============================================================
            # MÉTRICAS COMPLETAS
            # ============================================================
            self.txt_train.insert(tk.END, "📈 MÉTRICAS DE EVALUACIÓN:\n")

            # --- Entrenamiento ---
            self.txt_train.insert(tk.END, f"  ENTRENAMIENTO:\n")
            self.txt_train.insert(tk.END, f"    • EG:   {eval_train['EG']:.8f}\n")
            self.txt_train.insert(tk.END, f"    • MAE:  {eval_train['MAE']:.8f}\n")
            self.txt_train.insert(tk.END, f"    • RMSE: {eval_train['RMSE']:.8f}\n")

            # --- Prueba ---
            self.txt_train.insert(tk.END, f"\n  PRUEBA:\n")
            self.txt_train.insert(tk.END, f"    • EG:   {eval_test['EG']:.8f}\n")
            self.txt_train.insert(tk.END, f"    • MAE:  {eval_test['MAE']:.8f}\n")
            self.txt_train.insert(tk.END, f"    • RMSE: {eval_test['RMSE']:.8f}\n\n")

            # ============================================================
            # DETALLE DE SALIDAS
            # ============================================================
            self.txt_train.insert(tk.END, "🔎 DETALLE DE SALIDAS (Entrenamiento):\n")
            self.txt_train.insert(tk.END, f"{'Patrón':<10} {'Yd (Deseada)':<20} {'Yr (Predicha)':<20} {'Error Abs.':<15}\n")
            self.txt_train.insert(tk.END, "-"*70 + "\n")

            Yd_train = self.Y_train.flatten()
            Yr_train = eval_train["Yr"]
            for i in range(len(Yd_train)):
                err = abs(Yd_train[i] - Yr_train[i])
                self.txt_train.insert(tk.END, f"{i+1:<10} {Yd_train[i]:<20.8f} {Yr_train[i]:<20.8f} {err:<15.8f}\n")
            self.txt_train.insert(tk.END, "\n")

            self.txt_train.insert(tk.END, "🔎 DETALLE DE SALIDAS (Prueba):\n")
            self.txt_train.insert(tk.END, f"{'Patrón':<10} {'Yd (Real)':<20} {'Yr (Predicha)':<20} {'Error Abs.':<15}\n")
            self.txt_train.insert(tk.END, "-"*70 + "\n")

            Yd_test = self.Y_test.flatten()
            Yr_test = eval_test["Yr"]
            for i in range(len(Yd_test)):
                err = abs(Yd_test[i] - Yr_test[i])
                self.txt_train.insert(tk.END, f"{i+1:<10} {Yd_test[i]:<20.8f} {Yr_test[i]:<20.8f} {err:<15.8f}\n")
            self.txt_train.insert(tk.END, "\n")

            # ============================================================
            # CONVERGENCIA FINAL
            # ============================================================
            self.txt_train.insert(tk.END, "✅ PASO 10: Verificación de Convergencia\n")
            conv, msg = self.model.check_convergence(eval_train['EG'], self.error_opt_var.get())
            self.txt_train.insert(tk.END, f"  {msg}\n\n")

            self.txt_train.insert(tk.END, "📋 RESUMEN FINAL DEL MODELO:\n")
            self.txt_train.insert(tk.END, f"  • Centros radiales: {self.model.n_centers}\n")
            self.txt_train.insert(tk.END, f"  • Patrones de entrenamiento: {len(self.X_train)}\n")
            self.txt_train.insert(tk.END, f"  • Patrones de prueba: {len(self.X_test)}\n")
            self.txt_train.insert(tk.END, f"  • Error óptimo configurado: {self.error_opt_var.get()}\n")
            self.txt_train.insert(tk.END, f"  • Convergencia: {'✅ Sí' if conv else '❌ No'}\n\n")

            # ============================================================
            # PASO 8: Guardar resultados
            # ============================================================
            self.last_eval = {"train": eval_train, "test": eval_test}
            results = {
                "config": {
                    "n_centers": self.model.n_centers,
                    "train_pct": split_info['pct_train'],
                    "test_pct": split_info['pct_test'],
                    "error_opt": self.error_opt_var.get()
                },
                "centers": self.model.centers,
                "weights": self.model.weights,
                "matriz_A": A,
                "EG_train": eval_train["EG"],
                "MAE_train": eval_train["MAE"],
                "RMSE_train": eval_train["RMSE"],
                "convergencia_train": conv,
                "EG_test": eval_test["EG"],
                "MAE_test": eval_test["MAE"],
                "RMSE_test": eval_test["RMSE"]
            }

            json_path, csv_path = save_results(results)
            self.txt_train.insert(tk.END, f"💾 Resultados guardados:\n")
            self.txt_train.insert(tk.END, f"  • JSON: {json_path}\n")
            self.txt_train.insert(tk.END, f"  • CSV:  {csv_path}\n\n")

            self.txt_train.insert(tk.END, "=" * 80 + "\n")
            self.txt_train.insert(tk.END, "✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE\n")
            self.txt_train.insert(tk.END, "=" * 80 + "\n")
            
            # Activar botón de visualización de matriz
            self.A = A  # Guardamos la matriz en el objeto para usarla luego
            self.cols_header = cols_header
            self.btn_show_matrix.config(state="normal")

            messagebox.showinfo("Éxito", "Modelo entrenado correctamente")

        except Exception as e:
            messagebox.showerror("Error", f"Error durante entrenamiento:\n{str(e)}")
            import traceback
            traceback.print_exc()
                

    def upload_model_to_drive(self):
        if not Pydrive_available:
            messagebox.showerror("Error", "pydrive2 no está instalado")
            return
        
        # Buscar último modelo guardado
        model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".json")]
        if not model_files:
            messagebox.showwarning("Aviso", "No hay modelos para subir")
            return
        
        try:
            for f in model_files[:3]:  # Subir últimos 3 archivos
                path = os.path.join(MODELS_DIR, f)
                msg = upload_to_drive(path)
                print(msg)
            
            messagebox.showinfo("Drive", "✅ Archivos subidos a Google Drive")
        except Exception as e:
            messagebox.showerror("Error Drive", str(e))

    # ==================== TAB 4: SIMULACIÓN ====================
    def _build_tab_sim(self):
        f = self.tab_sim
        ttk.Label(f, text="Simulación y Predicción", style="Title.TLabel").pack(pady=10)

        # ------------------------------------------------------------
        # Botón para cargar modelo
        # ------------------------------------------------------------
        frm_top = ttk.Frame(f)
        frm_top.pack(fill="x", padx=20, pady=10)

        ttk.Button(
            frm_top,
            text="📂 Cargar Modelo (JSON/CSV)",
            command=self.load_model_dialog,
            width=25
        ).pack(side="left", padx=5)

        # ------------------------------------------------------------
        # Simulación manual
        # ------------------------------------------------------------
        frm_manual = ttk.LabelFrame(f, text="✏️ Simulación Manual", padding=15)
        frm_manual.pack(fill="x", padx=20, pady=10)

        ttk.Label(frm_manual, text="Entrada (valores separados por comas):").pack(anchor="w", pady=5)
        self.sim_entry = ttk.Entry(frm_manual, width=80)
        self.sim_entry.pack(fill="x", pady=5)

        ttk.Button(
            frm_manual,
            text="🔍 Predecir",
            command=self.simulate_manual
        ).pack(pady=5)

        # ------------------------------------------------------------
        # Resultados
        # ------------------------------------------------------------
        ttk.Label(f, text="📋 Resultados de Simulación:").pack(anchor="w", padx=20, pady=(10, 5))
        self.txt_sim = tk.Text(f, height=20, width=120, font=("Consolas", 9))
        self.txt_sim.pack(padx=20, pady=5, fill="both", expand=True)

    def simulate_manual(self):
        if self.model is None or not self.model.trained:
            messagebox.showwarning("Aviso", "Cargue o entrene un modelo primero")
            return
        
        s = self.sim_entry.get().strip()
        if not s:
            return
        
        try:
            arr = np.array([float(x.strip()) for x in s.split(",")]).reshape(1, -1)
            
            if self.scaler is not None:
                arr = self.scaler.transform(arr)
            
            ypred = self.model.predict(arr)[0]
            
            self.txt_sim.insert(tk.END, f"🔍 Entrada: {s}\n")
            self.txt_sim.insert(tk.END, f"📤 Predicción: {float(ypred):.8f}\n")
            self.txt_sim.insert(tk.END, "-"*80 + "\n")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            
    def save_model_dialog(self):
            """Guarda el modelo entrenado en un solo archivo JSON o CSV completo (centros, pesos, métricas, dataset, etc.)"""
            if self.model is None or not getattr(self.model, "trained", False):
                messagebox.showwarning("Aviso", "⚠️ No hay modelo entrenado para guardar.")
                return

            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[
                    ("Modelo completo JSON", "*.json"),
                    ("Modelo completo CSV", "*.csv"),
                ],
                initialfile="modelo_completo_rbf"
            )

            if not file_path:
                return

            try:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                ext = os.path.splitext(file_path)[1].lower()
                os.makedirs(MODELS_DIR, exist_ok=True)

                # 🔹 Conversión segura a tipos serializables
                def safe_convert(obj):
                    import numpy as np
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, (np.float32, np.float64)):
                        return float(obj)
                    elif isinstance(obj, (np.int32, np.int64)):
                        return int(obj)
                    elif isinstance(obj, dict):
                        return {k: safe_convert(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [safe_convert(x) for x in obj]
                    else:
                        return obj

                # Obtener matriz A si existe
                A_path = os.path.join(RESULTS_DIR, "matriz_A.csv")
                A = pd.read_csv(A_path).values if os.path.exists(A_path) else None

                # Datos completos del modelo
                full_data = {
                    "config": {
                        "n_centers": int(getattr(self.model, "n_centers", 0)),
                        "error_opt": float(self.error_opt_var.get()),
                        "train_pct": self.train_pct_var.get(),
                    },
                    "centers": safe_convert(self.model.centers) if self.model.centers is not None else [],
                    "weights": safe_convert(self.model.weights) if self.model.weights is not None else [],
                    "matriz_A": safe_convert(A) if A is not None else [],
                    "scaler_mean": safe_convert(self.scaler.mean_) if self.scaler else None,
                    "scaler_std": safe_convert(self.scaler.scale_) if self.scaler else None,
                    "meta": safe_convert(self.meta),
                    "train": safe_convert(self.last_eval.get("train", {})) if self.last_eval else {},
                    "test": safe_convert(self.last_eval.get("test", {})) if self.last_eval else {}
                }

                # ---------------------- GUARDAR JSON ----------------------
                if ext == ".json":
                    with open(file_path, "w", encoding="utf8") as f:
                        json.dump(full_data, f, indent=2, ensure_ascii=False)
                    messagebox.showinfo("Guardado", f"✅ Modelo completo guardado en:\n{file_path}")

                # ---------------------- GUARDAR CSV ----------------------
                elif ext == ".csv":
                    flat_data = []
                    for k, v in full_data.items():
                        if isinstance(v, dict):
                            for subk, subv in v.items():
                                flat_data.append({"Sección": k, "Campo": subk, "Valor": json.dumps(subv)})
                        else:
                            flat_data.append({"Sección": "root", "Campo": k, "Valor": json.dumps(v)})

                    pd.DataFrame(flat_data).to_csv(file_path, index=False)
                    messagebox.showinfo("Guardado", f"✅ Modelo completo guardado en:\n{file_path}")

                else:
                    messagebox.showerror("Error", "Formato no soportado (use .json o .csv).")

            except Exception as e:
                import traceback; traceback.print_exc()
                messagebox.showerror("Error", f"Error al guardar modelo:\n{str(e)}")


    def load_model_dialog(self):
        """Carga un modelo completo (JSON o CSV) y habilita simulación sin reentrenar."""
        path = filedialog.askopenfilename(
            title="Seleccionar modelo guardado",
            filetypes=[("Modelos RBF", "*.json *.csv")],
            initialdir="modelos"
        )

        if not path:
            return

        try:
            ext = os.path.splitext(path)[1].lower()

            # ---------------------- CARGAR JSON ----------------------
            if ext == ".json":
                with open(path, "r", encoding="utf8") as f:
                    data = json.load(f)

            # ---------------------- CARGAR CSV ----------------------
            elif ext == ".csv":
                df = pd.read_csv(path)
                data = {}
                for _, row in df.iterrows():
                    section = row["Sección"]
                    field = row["Campo"]
                    value = json.loads(row["Valor"])
                    if section not in data:
                        data[section] = {}
                    data[section][field] = value
            else:
                messagebox.showerror("Error", "Formato no soportado.")
                return

            # ---------------------- RECONSTRUIR MODELO ----------------------
            from .model_rbf import RBFModel
            self.model = RBFModel(n_centers=int(data["config"]["n_centers"]))
            self.model.centers = np.array(data["centers"])
            self.model.weights = np.array(data["weights"]).flatten()
            self.model.trained = True

            # Escalador
            self.scaler = None
            if data.get("scaler_mean") and data.get("scaler_std"):
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                self.scaler.mean_ = np.array(data["scaler_mean"])
                self.scaler.scale_ = np.array(data["scaler_std"])
                self.scaler.n_features_in_ = len(self.scaler.mean_)

            # Metadatos y evaluación
            self.meta = data.get("meta", {})
            self.last_eval = {"train": data.get("train", {}), "test": data.get("test", {})}

            # Activar simulación sin entrenamiento
            if hasattr(self, "btn_predict_manual"):
                self.btn_predict_manual.config(state="normal")
            if hasattr(self, "btn_simulate_test"):
                self.btn_simulate_test.config(state="normal")

            # Mostrar resumen
            info = f"✅ Modelo cargado: {os.path.basename(path)}\n"
            info += f"• Centros radiales: {self.model.n_centers}\n"
            info += f"• Pesos: {len(self.model.weights)}\n"
            if self.last_eval:
                tr = self.last_eval["train"]
                ts = self.last_eval["test"]
                info += "\n📈 MÉTRICAS:\n"
                info += f"  Entrenamiento → EG={tr.get('EG', 0):.6f}, MAE={tr.get('MAE', 0):.6f}, RMSE={tr.get('RMSE', 0):.6f}\n"
                info += f"  Prueba        → EG={ts.get('EG', 0):.6f}, MAE={ts.get('MAE', 0):.6f}, RMSE={ts.get('RMSE', 0):.6f}\n"

            if hasattr(self, "lbl_model_info"):
                self.lbl_model_info.config(text=info)

            messagebox.showinfo("Modelo cargado", f"✅ Modelo cargado correctamente:\n{os.path.basename(path)}\n\nYa puedes simular sin reentrenar.")

        except Exception as e:
            import traceback; traceback.print_exc()
            messagebox.showerror("Error", f"No se pudo cargar el modelo:\n{str(e)}")

    

    # ==================== TAB 5: GRÁFICAS ====================
    def _build_tab_graph(self):
        f = self.tab_graph
        
        ttk.Label(f, text="Visualización de Resultados", style="Title.TLabel").pack(pady=10)
        
        ttk.Button(f, text="📊 Generar Gráficas", 
                  command=self.show_graphs, width=20).pack(pady=10)
        
        self.graph_frame = ttk.Frame(f)
        self.graph_frame.pack(fill="both", expand=True, padx=10, pady=10)

    def show_graphs(self):
        if self.model is None or self.last_eval is None:
            messagebox.showwarning("Aviso", "Entrene el modelo primero")
            return
        
        try:
            # Limpiar frame
            for widget in self.graph_frame.winfo_children():
                widget.destroy()
            
            # Obtener datos
            eval_train = self.last_eval["train"]
            eval_test = self.last_eval["test"]
            
            Yd_train = self.Y_train.flatten()
            Yr_train = eval_train["Yr"]
            Yd_test = self.Y_test.flatten()
            Yr_test = eval_test["Yr"]
            
            # Crear figura con 4 gráficas (2x2)
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Análisis de la Red RBF', fontsize=14, fontweight='bold')
            
            # Gráfica 1: Yd vs Yr (Entrenamiento y Prueba)
            ax1 = axs[0, 0]
            idx_train = range(1, len(Yd_train) + 1)
            idx_test = range(len(Yd_train) + 1, len(Yd_train) + len(Yd_test) + 1)
            
            ax1.plot(idx_train, Yd_train, 'o-', label='Yd Train', color='blue', markersize=4)
            ax1.plot(idx_train, Yr_train, 's--', label='Yr Train', color='cyan', markersize=3)
            ax1.plot(idx_test, Yd_test, 'o-', label='Yd Test', color='red', markersize=4)
            ax1.plot(idx_test, Yr_test, 's--', label='Yr Test', color='orange', markersize=3)
            ax1.set_xlabel('Patrón')
            ax1.set_ylabel('Salida')
            ax1.set_title('Yd vs Yr (Entrenamiento y Prueba)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gráfica 2: Error por Patrón
            ax2 = axs[0, 1]
            errors_train = eval_train["errors_per_pattern"]
            errors_test = eval_test["errors_per_pattern"]
            
            ax2.bar(idx_train, errors_train, label='Error Train', color='lightblue', alpha=0.7)
            ax2.bar(idx_test, errors_test, label='Error Test', color='lightcoral', alpha=0.7)
            ax2.axhline(y=self.error_opt_var.get(), color='green', linestyle='--', 
                       label=f'Error Óptimo ({self.error_opt_var.get()})')
            ax2.set_xlabel('Patrón')
            ax2.set_ylabel('Error Absoluto')
            ax2.set_title('Error por Patrón')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Gráfica 3: Dispersión Yd vs Yr
            ax3 = axs[1, 0]
            ax3.scatter(Yd_train, Yr_train, alpha=0.6, label='Entrenamiento', color='blue', s=30)
            ax3.scatter(Yd_test, Yr_test, alpha=0.6, label='Prueba', color='red', s=30)
            
            # Línea diagonal ideal (Yd = Yr)
            all_vals = np.concatenate([Yd_train, Yr_train, Yd_test, Yr_test])
            min_val = all_vals.min()
            max_val = all_vals.max()
            ax3.plot([min_val, max_val], [min_val, max_val], 'g--', 
                    linewidth=2, label='Ideal (Yd=Yr)')
            
            ax3.set_xlabel('Yd (Valor Real)')
            ax3.set_ylabel('Yr (Predicción)')
            ax3.set_title('Dispersión: Yd vs Yr')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Gráfica 4: Comparación de Métricas
            ax4 = axs[1, 1]
            metrics = ['EG', 'MAE', 'RMSE']
            train_vals = [eval_train['EG'], eval_train['MAE'], eval_train['RMSE']]
            test_vals = [eval_test['EG'], eval_test['MAE'], eval_test['RMSE']]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax4.bar(x - width/2, train_vals, width, label='Entrenamiento', color='skyblue')
            ax4.bar(x + width/2, test_vals, width, label='Prueba', color='salmon')
            ax4.axhline(y=self.error_opt_var.get(), color='green', linestyle='--', 
                       label=f'Óptimo ({self.error_opt_var.get()})')
            
            ax4.set_xlabel('Métrica')
            ax4.set_ylabel('Valor')
            ax4.set_title('Comparación de Métricas (Train vs Test)')
            ax4.set_xticks(x)
            ax4.set_xticklabels(metrics)
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            # Integrar en Tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
            # Guardar gráficas
            fig.savefig(os.path.join(RESULTS_DIR, "graficas_rbf.png"), dpi=150, bbox_inches='tight')
            messagebox.showinfo("Gráficas", "✅ Gráficas generadas y guardadas en resultados/")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar gráficas:\n{str(e)}")
            import traceback
            traceback.print_exc()

    # ==================== TAB 6: RESULTADOS ====================
    def _build_tab_results(self):
        f = self.tab_results
        
        ttk.Label(f, text="Resultados y Trazabilidad", style="Title.TLabel").pack(pady=10)
        
        # Botones
        frm_btns = ttk.Frame(f)
        frm_btns.pack(fill="x", padx=20, pady=10)
        
        ttk.Button(frm_btns, text="📄 Mostrar último resultado", 
                  command=self.show_last_results).pack(side="left", padx=5)
        ttk.Button(frm_btns, text="☁️ Subir resultados a Drive", 
                  command=self.upload_results_to_drive).pack(side="left", padx=5)
        ttk.Button(frm_btns, text="📋 Generar reporte completo", 
                  command=self.generate_full_report).pack(side="left", padx=5)
        
        # Tabla de métricas
        frm_table = ttk.LabelFrame(f, text="📊 Tabla de Métricas", padding=10)
        frm_table.pack(fill="x", padx=20, pady=10)
        
        cols = ("conjunto", "eg", "mae", "rmse", "convergencia")
        self.metrics_table = ttk.Treeview(frm_table, columns=cols, show="headings", height=3)
        self.metrics_table.heading("conjunto", text="Conjunto")
        self.metrics_table.heading("eg", text="EG")
        self.metrics_table.heading("mae", text="MAE")
        self.metrics_table.heading("rmse", text="RMSE")
        self.metrics_table.heading("convergencia", text="Convergencia")
        
        for col in cols:
            self.metrics_table.column(col, width=150, anchor="center")
        
        self.metrics_table.pack(fill="x")
        
        # Área de texto para resultados completos
        ttk.Label(f, text="📋 Resultados Detallados:").pack(anchor="w", padx=20, pady=(10,5))
        self.results_text = tk.Text(f, height=20, width=120, font=("Consolas", 9))
        self.results_text.pack(padx=20, pady=5, fill="both", expand=True)

    def show_last_results(self):
        json_path = os.path.join(RESULTS_DIR, "resultados_rbf.json")
        
        if not os.path.exists(json_path):
            messagebox.showinfo("Resultados", "No hay resultados disponibles. Entrene el modelo primero.")
            return
        
        try:
            with open(json_path, "r", encoding="utf8") as f:
                data = json.load(f)
            
            # Actualizar tabla de métricas
            for item in self.metrics_table.get_children():
                self.metrics_table.delete(item)
            
            # Insertar datos de entrenamiento
            conv_train = "✅ Sí" if data.get("convergencia_train", False) else "❌ No"
            self.metrics_table.insert("", "end", values=(
                "Entrenamiento",
                f"{data['EG_train']:.6f}",
                f"{data['MAE_train']:.6f}",
                f"{data['RMSE_train']:.6f}",
                conv_train
            ))
            
            # Insertar datos de prueba
            self.metrics_table.insert("", "end", values=(
                "Prueba",
                f"{data['EG_test']:.6f}",
                f"{data['MAE_test']:.6f}",
                f"{data['RMSE_test']:.6f}",
                "—"
            ))
            
            # Mostrar resultados completos
            self.results_text.delete("1.0", tk.END)
            self.results_text.insert(tk.END, "="*80 + "\n")
            self.results_text.insert(tk.END, "RESULTADOS COMPLETOS DEL ENTRENAMIENTO RBF\n")
            self.results_text.insert(tk.END, "="*80 + "\n\n")
            
            self.results_text.insert(tk.END, "📋 CONFIGURACIÓN:\n")
            if "config" in data:
                for k, v in data["config"].items():
                    self.results_text.insert(tk.END, f"  • {k}: {v}\n")
            self.results_text.insert(tk.END, "\n")
            
            self.results_text.insert(tk.END, "🎯 CENTROS RADIALES:\n")
            if "centers" in data and data["centers"]:
                for i, center in enumerate(data["centers"], 1):
                    center_str = ", ".join([f"{x:.6f}" for x in center])
                    self.results_text.insert(tk.END, f"  Centro {i}: [{center_str}]\n")
            self.results_text.insert(tk.END, "\n")
            
            self.results_text.insert(tk.END, "⚖️ PESOS DE LA RED:\n")
            if "weights" in data and data["weights"]:
                self.results_text.insert(tk.END, f"  W₀ (bias): {data['weights'][0]:.8f}\n")
                for i, w in enumerate(data["weights"][1:], 1):
                    self.results_text.insert(tk.END, f"  W{i}: {w:.8f}\n")
            self.results_text.insert(tk.END, "\n")
            
            self.results_text.insert(tk.END, "📊 MÉTRICAS DE EVALUACIÓN:\n")
            self.results_text.insert(tk.END, f"  ENTRENAMIENTO:\n")
            self.results_text.insert(tk.END, f"    • EG:   {data['EG_train']:.8f}\n")
            self.results_text.insert(tk.END, f"    • MAE:  {data['MAE_train']:.8f}\n")
            self.results_text.insert(tk.END, f"    • RMSE: {data['RMSE_train']:.8f}\n")
            self.results_text.insert(tk.END, f"    • Convergencia: {conv_train}\n\n")
            
            self.results_text.insert(tk.END, f"  PRUEBA:\n")
            self.results_text.insert(tk.END, f"    • EG:   {data['EG_test']:.8f}\n")
            self.results_text.insert(tk.END, f"    • MAE:  {data['MAE_test']:.8f}\n")
            self.results_text.insert(tk.END, f"    • RMSE: {data['RMSE_test']:.8f}\n\n")
            
            self.results_text.insert(tk.END, "="*80 + "\n")
            
            messagebox.showinfo("Resultados", "✅ Resultados cargados correctamente")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar resultados:\n{str(e)}")

    def upload_results_to_drive(self):
        if not Pydrive_available:
            messagebox.showerror("Error", "pydrive2 no está instalado")
            return
        
        try:
            files_to_upload = [
                os.path.join(RESULTS_DIR, "resultados_rbf.json"),
                os.path.join(RESULTS_DIR, "resultados_rbf_train.csv"),
                os.path.join(RESULTS_DIR, "resultados_rbf_test.csv"),
                os.path.join(RESULTS_DIR, "graficas_rbf.png")
            ]
            
            uploaded = []
            for path in files_to_upload:
                if os.path.exists(path):
                    msg = upload_to_drive(path)
                    uploaded.append(msg)
            
            if uploaded:
                messagebox.showinfo("Drive", "\n".join(uploaded))
            else:
                messagebox.showwarning("Drive", "No hay archivos para subir")
                
        except Exception as e:
            messagebox.showerror("Error Drive", str(e))

    def generate_full_report(self):
        """Genera un reporte completo en formato TXT"""
        if self.model is None or self.last_eval is None:
            messagebox.showwarning("Aviso", "Entrene el modelo primero")
            return
        
        try:
            report_path = os.path.join(RESULTS_DIR, "reporte_completo_rbf.txt")
            
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("="*80 + "\n")
                f.write("REPORTE COMPLETO - RED NEURONAL RBF\n")
                f.write("Examen Práctico - Inteligencia Artificial\n")
                f.write("="*80 + "\n\n")
                
                f.write("OBJETIVO:\n")
                f.write("Desarrollar un aplicativo funcional que implemente el entrenamiento,\n")
                f.write("simulación, evaluación y validación de una red neuronal RBF.\n\n")
                
                f.write("="*80 + "\n")
                f.write("1. INFORMACIÓN DEL DATASET\n")
                f.write("="*80 + "\n")
                if self.meta:
                    f.write(f"  • Total de patrones: {self.meta['n_patterns']}\n")
                    f.write(f"  • Número de entradas: {self.meta['n_inputs']}\n")
                    f.write(f"  • Número de salidas: {self.meta['n_outputs']}\n")
                    f.write(f"  • Columnas de entrada: {self.meta['X_cols']}\n")
                    f.write(f"  • Columna de salida: {self.meta['Y_col']}\n\n")
                
                f.write("="*80 + "\n")
                f.write("2. PREPROCESAMIENTO\n")
                f.write("="*80 + "\n")
                f.write("  • Normalización: Z-score (estandarización)\n")
                f.write("  • Tratamiento de valores faltantes: Eliminación\n\n")
                
                f.write("="*80 + "\n")
                f.write("3. PARTICIÓN DEL DATASET\n")
                f.write("="*80 + "\n")
                if self.X_train is not None:
                    train_pct = (len(self.X_train) / len(self.X)) * 100
                    test_pct = 100 - train_pct
                    f.write(f"  • Entrenamiento: {len(self.X_train)} patrones ({train_pct:.1f}%)\n")
                    f.write(f"  • Prueba: {len(self.X_test)} patrones ({test_pct:.1f}%)\n\n")
                
                f.write("="*80 + "\n")
                f.write("4. CONFIGURACIÓN DE LA RED RBF\n")
                f.write("="*80 + "\n")
                f.write(f"  • Número de centros radiales: {self.model.n_centers}\n")
                f.write(f"  • Función de activación: FA(d) = d² * ln(d)\n")
                f.write(f"  • Error óptimo (EG): {self.error_opt_var.get()}\n")
                f.write(f"  • Método de selección: Aleatorio dentro del rango\n\n")
                
                f.write("="*80 + "\n")
                f.write("5. CENTROS RADIALES FINALES\n")
                f.write("="*80 + "\n")
                if self.model.centers is not None:
                    for i, center in enumerate(self.model.centers, 1):
                        center_str = ", ".join([f"{x:.6f}" for x in center])
                        f.write(f"  Centro {i}: [{center_str}]\n")
                f.write("\n")
                
                f.write("="*80 + "\n")
                f.write("6. PESOS DE LA RED (W)\n")
                f.write("="*80 + "\n")
                if self.model.weights is not None:
                    f.write(f"  W₀ (bias):  {self.model.weights[0]:.8f}\n")
                    for i, w in enumerate(self.model.weights[1:], 1):
                        f.write(f"  W{i}:        {w:.8f}\n")
                f.write("\n")
                
                f.write("="*80 + "\n")
                f.write("7. MÉTRICAS DE EVALUACIÓN\n")
                f.write("="*80 + "\n\n")
                
                eval_train = self.last_eval["train"]
                eval_test = self.last_eval["test"]
                
                f.write("  CONJUNTO DE ENTRENAMIENTO:\n")
                f.write(f"    • EG (Error General):           {eval_train['EG']:.8f}\n")
                f.write(f"    • MAE (Error Absoluto Medio):   {eval_train['MAE']:.8f}\n")
                f.write(f"    • RMSE (Raíz Error Cuadrático): {eval_train['RMSE']:.8f}\n")
                
                conv, msg = self.model.check_convergence(eval_train['EG'], self.error_opt_var.get())
                f.write(f"    • Convergencia: {msg}\n\n")
                
                f.write("  CONJUNTO DE PRUEBA:\n")
                f.write(f"    • EG (Error General):           {eval_test['EG']:.8f}\n")
                f.write(f"    • MAE (Error Absoluto Medio):   {eval_test['MAE']:.8f}\n")
                f.write(f"    • RMSE (Raíz Error Cuadrático): {eval_test['RMSE']:.8f}\n\n")
                
                f.write("="*80 + "\n")
                f.write("8. TABLA RESUMEN DE MÉTRICAS\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"{'Conjunto':<20} {'EG':<15} {'MAE':<15} {'RMSE':<15} {'Convergencia':<15}\n")
                f.write("-"*80 + "\n")
                f.write(f"{'Entrenamiento':<20} {eval_train['EG']:<15.6f} {eval_train['MAE']:<15.6f} {eval_train['RMSE']:<15.6f} {'Sí' if conv else 'No':<15}\n")
                f.write(f"{'Prueba':<20} {eval_test['EG']:<15.6f} {eval_test['MAE']:<15.6f} {eval_test['RMSE']:<15.6f} {'—':<15}\n\n")
                
                f.write("="*80 + "\n")
                f.write("9. VERIFICACIÓN DE CONVERGENCIA\n")
                f.write("="*80 + "\n")
                f.write(f"  Error General (Entrenamiento): {eval_train['EG']:.8f}\n")
                f.write(f"  Error Óptimo configurado: {self.error_opt_var.get()}\n")
                f.write(f"  Resultado: {msg}\n\n")
                
                if not conv:
                    f.write("  RECOMENDACIÓN: Aumentar el número de centros radiales\n")
                    f.write("  y repetir el entrenamiento para mejorar la convergencia.\n\n")
                
                f.write("="*80 + "\n")
                f.write("10. ARCHIVOS GENERADOS\n")
                f.write("="*80 + "\n")
                f.write("  • resultados_rbf.json - Resultados completos en JSON\n")
                f.write("  • resultados_rbf_train.csv - Predicciones de entrenamiento\n")
                f.write("  • resultados_rbf_test.csv - Predicciones de prueba\n")
                f.write("  • graficas_rbf.png - Visualizaciones gráficas\n")
                f.write("  • modelo_rbf.json - Modelo entrenado (configuración)\n")
                f.write("  • modelo_rbf_centros.csv - Centros radiales\n")
                f.write("  • modelo_rbf_pesos.csv - Pesos de la red\n\n")
                
                f.write("="*80 + "\n")
                f.write("CONCLUSIÓN\n")
                f.write("="*80 + "\n")
                f.write("El modelo RBF ha sido entrenado exitosamente siguiendo todos los\n")
                f.write("pasos del examen práctico de Inteligencia Artificial. Se han calculado\n")
                f.write("las métricas de evaluación (EG, MAE, RMSE) y se ha verificado la\n")
                f.write("convergencia del modelo según el error óptimo establecido.\n\n")
                
                if conv:
                    f.write("✅ El modelo CONVERGE satisfactoriamente.\n")
                else:
                    f.write("⚠️ El modelo NO CONVERGE. Se recomienda ajustar los parámetros.\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("FIN DEL REPORTE\n")
                f.write("="*80 + "\n")
            
            messagebox.showinfo("Reporte", f"✅ Reporte completo generado:\n{report_path}")
            
            # Abrir reporte en el visor
            self.results_text.delete("1.0", tk.END)
            with open(report_path, "r", encoding="utf-8") as f:
                self.results_text.insert(tk.END, f.read())
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar reporte:\n{str(e)}")