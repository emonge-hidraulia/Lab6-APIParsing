# Lab 6: Parsing de Respuestas de APIs, Serialización y Resiliencia

**Asignatura:** Programación de Inteligencia Artificial  
**Especialización:** FP Grado Superior en Inteligencia Artificial & Big Data  
**Sesión:** 08 - Parsing de Respuestas de APIs, Serialización y Deserialización de Objetos de Modelo  

---

## 📋 Descripción del Proyecto

Este proyecto implementa un cliente de ingesta resiliente y tipado en Python para APIs de Inteligencia Artificial Generativa (Google Gemini / AI Studio).

### Características Principales

1. **Modelos Fuertes con Pydantic v2**:
   - `AnalisisResena`: Valida estrictamente la inferencia devuelta por el modelo (`sentiment`, `score` acotado entre `0.0` y `1.0`, `tema`, `resumen`, `recommendation`).
   - `ExpedienteConsolidado`: Agrupa el prompt original, el objeto validado y metadatos técnicos de ejecución.

2. **Tolerancia a Fallos y Exponential Backoff (Tenacity)**:
   - Detección selectiva de errores transitorios con `TransientAPIError` (HTTP 429 Rate Limit, 502, 503, 504 y timeouts de red).
   - Reintentos exponenciales automáticos: `stop_after_attempt(4)` con `wait_exponential(min=2, max=10)`.

3. **Ingesta por Lotes y Persistencia**:
   - Procesa un conjunto de reseñas heterogéneas.
   - Exporta los resultados limpios e indentados a `resultados_ingesta.json`.

4. **Modo Híbrido**:
   - Conexión real con la API REST de Google Gemini si se provee la variable `GEMINI_API_KEY`.
   - Motor de parsing semántico determinista integrado para pruebas sin necesidad de cuota externa.

---

## 🚀 Instalación y Ejecución

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. (Opcional) Configurar API Key de Gemini
# Crear archivo .env con: GEMINI_API_KEY=tu_api_key_aqui

# 3. Ejecutar el script principal
python Lab6_APIParsing_EmilioMonge.py
```

---

## 📁 Estructura del Repositorio

- `Lab6_APIParsing_EmilioMonge.py`: Script principal con la implementación completa de Lab 6.
- `resultados_ingesta.json`: Salida generada con los expedientes analizados y serializados.
- `requirements.txt`: Dependencias del proyecto.
- `sesion_08.html`: Presentación y especificaciones de la sesión.
