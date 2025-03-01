# Guía de Gestión de Modelos

## Estructura de Directorios
```
models/
├── active/              # Modelos en uso actual
├── archive/            # Versiones históricas
└── metadata/           # Configuración y métricas
```

## Tipos de Modelos
1. **Predictor Corto Plazo**
   - Predicciones 1-5 minutos
   - Mayor frecuencia actualización
   - Usa datos más recientes

2. **Predictor Medio Plazo**
   - Predicciones 5-30 minutos
   - Balance entre velocidad y precisión
   - Usa indicadores técnicos

3. **Predictor Largo Plazo**
   - Predicciones 30+ minutos
   - Enfoque en análisis tendencial
   - Usa datos multitemporales

## Mantenimiento de Modelos

### Entrenamiento de Nuevos Modelos
```bash
# Entrenar todos los tipos
python main.py --train-only

# Entrenar tipo específico
python main.py --train-only --model-type short_term
```

### Gestión de Archivos de Modelo
```bash
# Ver estadísticas
python clean_models.py --action stats

# Archivar modelos antiguos
python clean_models.py --action archive --keep 5

# Limpiar archivos expirados
python clean_models.py --action clean --days 30
```

### Validación de Modelos
Antes del despliegue, los modelos se validan contra:
- Umbral mínimo de precisión
- Desviación máxima predicción
- Métricas históricas rendimiento
