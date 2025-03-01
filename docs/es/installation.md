# Guía de Instalación

## Requisitos del Sistema
- Python 3.11 o superior
- Conexión a internet estable (mínimo 1 Mbps)
- Cuenta activa en Deriv.com
- VS Code (recomendado)

## Pasos de Instalación

### 1. Clonar el Repositorio
```bash
git clone https://github.com/your-username/deriv-ml-trading-bot.git
cd deriv-ml-trading-bot
```

### 2. Configurar Entorno Virtual
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 4. Configuración del Entorno
Copiar `.env.example` a `.env`:
```bash
# Windows
copy .env.example .env

# macOS/Linux
cp .env.example .env
```

Editar `.env` con tus tokens API:
```ini
DERIV_API_TOKEN_DEMO=tu_token_demo
DERIV_API_TOKEN_REAL=tu_token_real
DERIV_BOT_ENV=demo
APP_ID=1089
```

### 5. Verificar Instalación
```bash
python test_api_connectivity.py
```

Para la configuración detallada de VS Code, consulta la [Guía de Configuración de VS Code](vscode_setup.md).
