# Brain AI - Asistente de PDFs con RAG

Aplicacion para hacer preguntas sobre documentos PDF usando RAG (Retrieval-Augmented Generation) con LangChain, OpenAI embeddings y ChromaDB.

modo de uso:
- Interfaz web con Streamlit en `interface.py`

## Que hace este proyecto

1. Carga un PDF
2. Lo divide en fragmentos (chunks)
3. Crea embeddings con OpenAI
4. Guarda/consulta vectores en Chroma
5. Responde preguntas con contexto recuperado del documento

## Requisitos

- Python 3.12 recomendado
- Cuenta de OpenAI con API key
- Entorno virtual (`.venv`)

## Instalacion

### 1. Crear y activar entorno virtual (Windows PowerShell)

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 2. Instalar dependencias

Si ya tienes `requirements.txt`:

```powershell
pip install -r requirements.txt
```
## Configuracion

Crea un archivo `.env` en la raiz del proyecto con:

```env
OPENAI_API_KEY=tu_api_key_aqui
```



## Ejecutar la version web (Streamlit)

```powershell
streamlit run interface.py
```

Luego abre la URL local que muestra Streamlit (normalmente http://localhost:8501), sube un PDF y empieza a chatear.

## Estructura del proyecto
- `interface.py`: interfaz web con Streamlit
- `requirements.txt`: dependencias congeladas
- `chroma_db/`: persistencia local de vectores (creada al procesar documentos)
- `.env`: variables de entorno (API key)

## Errores comunes

### 1) Pylance: "Import 'langchain.chains' could not be resolved"

Con LangChain 1.x, `RetrievalQA` se importa desde `langchain_classic.chains`, no desde `langchain.chains`.

Import correcto:

```python
from langchain_classic.chains import RetrievalQA
```

### 2) OPENAI_API_KEY no encontrada

- Verifica que `.env` exista en la raiz
- Verifica que tenga `OPENAI_API_KEY=...`
- Reinicia terminal o la app tras cambiar `.env`

### 3) Warning de Streamlit "missing ScriptRunContext"

Es normal si importas `interface.py` fuera de `streamlit run`.
Para uso normal, ejecuta siempre:

```powershell
streamlit run interface.py
```

## Mejoras recomendadas

- Persistir y reutilizar la base vectorial en Streamlit (`persist_directory`)
- Permitir elegir modelo por variable de entorno
- Agregar historial conversacional con memoria por sesion
- Añadir tests basicos de carga de PDF e indexado

