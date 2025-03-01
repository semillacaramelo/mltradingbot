# Guía Rápida

## Requisitos Previos
- Python 3.11 o superior
- Cuenta en Deriv.com
- Tokens API (demo/real)

## Instalación

1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/trading_bot_replit_v4.git
cd trading_bot_replit_v4
```

2. Ejecutar script de configuración
```bash
python environment_setup.py
```

3. Probar conectividad API
```bash
python test_api_connectivity.py
```

## Configuración

1. Obtener tokens API de Deriv.com:
   - Ve a [Página de Tokens API Deriv](https://app.deriv.com/account/api-token)
   - Crea tokens con los permisos necesarios:
     - Lectura
     - Trading
     - Pagos
     - Administración

2. Configurar el entorno:
   - Edita el archivo `.env`
   - Añade tus tokens API
   - Establece `DERIV_BOT_ENV=demo` para pruebas

## Uso

### Modo Demo
```bash
python main.py --env demo
```

### Modo Entrenamiento
```bash
python main.py --train-only
```

### Trading Real
```bash
# Primero establece en .env:
DERIV_BOT_ENV=real
DERIV_REAL_MODE_CONFIRMED=yes

# Luego ejecuta:
python main.py --env real
```

## Siguientes Pasos
- Revisa la [Guía de Configuración](configuration.md)
- Estudia la [Gestión de Riesgos](risk_management.md)
- Consulta [Solución de Problemas](troubleshooting.md)
