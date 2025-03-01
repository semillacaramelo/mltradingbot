# Guía de Configuración

## Variables de Entorno

### Variables Requeridas
- `DERIV_API_TOKEN_DEMO`: Token API de tu cuenta demo de Deriv
- `DERIV_API_TOKEN_REAL`: Token API de tu cuenta real de Deriv
- `DERIV_BOT_ENV`: Entorno de trading (`demo` o `real`)
- `APP_ID`: ID de aplicación Deriv (predeterminado: 1089)

### Variables de Seguridad
- `DERIV_REAL_MODE_CONFIRMED`: Debe ser "yes" para trading real

### Parámetros de Entrenamiento
- `SEQUENCE_LENGTH`: Longitud de secuencia para entrenamiento (predeterminado: 30)
- `TRAINING_EPOCHS`: Número de épocas de entrenamiento (predeterminado: 50)
- `MODEL_SAVE_PATH`: Directorio para guardar modelos (predeterminado: models)

## Archivo de Configuración (.env)

Ejemplo de configuración:
```ini
# Entorno de Trading
DERIV_BOT_ENV=demo

# Tokens API
DERIV_API_TOKEN_DEMO=tu_token_demo_aqui
DERIV_API_TOKEN_REAL=tu_token_real_aqui

# Configuración de Seguridad
DERIV_REAL_MODE_CONFIRMED=no

# Configuración de Aplicación
APP_ID=1089

# Parámetros de Entrenamiento
SEQUENCE_LENGTH=30
TRAINING_EPOCHS=50
MODEL_SAVE_PATH=models
```

## Notas de Seguridad
- Nunca subas tu archivo `.env` al repositorio
- Mantén los tokens API seguros
- Siempre prueba primero en modo demo
