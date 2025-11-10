# Entorno Virtual

## Información del Ambiente

- **Ubicación:** `./venv/`
- **Versión de Python:** 3.13.5
- **Sistema Operativo:** Windows PowerShell
- **Gestor de paquetes:** pip

## Activar el Virtual Environment

En PowerShell (Windows):
```powershell
.\venv\Scripts\Activate.ps1
```

En CMD (Windows):
```cmd
venv\Scripts\activate.bat
```

En Terminal/Bash (Linux/macOS):
```bash
source venv/bin/activate
```

## Instalar Dependencias

Una vez activado el virtual environment:
```bash
pip install -r requirements.txt
```

## Actualizar requirements.txt

Cuando agregues nuevas dependencias:
```bash
pip freeze > requirements.txt
```

## Desactivar el Virtual Environment

```bash
deactivate
```
