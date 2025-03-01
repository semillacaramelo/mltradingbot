# Guía de Conexión WebSocket

## Gestión de Conexiones

### Conexión Básica
Latido del Corazón
* Intervalo ping: 30 segundos
* Timeout conexión: 60 segundos
* Auto-reconexión: Habilitada

Recuperación de Errores

1. Pérdida de Conexión
* Reintentos con retroceso exponencial
* Máximo 5 intentos
* Seguimiento estado conexión
2. Límite de Tasa
* Máximo 60 peticiones/minuto
* Cola de peticiones
* Períodos de espera

### Tipos de Mensajes

Autenticación
```python
{
    "authorize": "TU_TOKEN_API",
    "req_id": "auth_1"
}
```
```python
{
    "ticks": "frxEURUSD",
    "subscribe": 1
}
```
```python
{
    "buy": 1,
    "parameters": {
        "amount": 10,
        "basis": "stake",
        "contract_type": "CALL", 
        "symbol": "frxEURUSD",
        "duration": 60
    }
}
```
