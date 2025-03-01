# Guía de Arquitectura de Modelos

## Estructura de Red Neuronal

### Capa de Entrada
- Longitud de secuencia: 30
- Características por punto temporal: 46
- Tamaño de lote: 32 (configurable)

### Capas LSTM
Capas Densas
```
Dense(32, activation='relu')
Dense(16, activation='relu')
Dense(1, activation='tanh')
```

### Tipos de Modelos
Modelo Corto Plazo
* 1-5 minutos predicción
* Mayor frecuencia actualización
* Enfocado en acción precio

Modelo Medio Plazo
* 5-30 minutos predicción
* Indicadores técnicos
* Regímenes de mercado

Modelo Largo Plazo
* 30+ minutos predicción
* Múltiples marcos temporales
* Análisis tendencial

### Archivos de Modelo
```plaintext
models/
├── short_term_AAAAMMDD_HHMMSS.keras
├── medium_term_AAAAMMDD_HHMMSS.keras
└── long_term_AAAAMMDD_HHMMSS.keras
```

### Configuración Entrenamiento
```python
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
```
