"""
=============================================================================
FP Grado Superior - Especialización en Inteligencia Artificial y Big Data
Módulo: Programación de Inteligencia Artificial
Sesión 8: Parsing de Respuestas de APIs, Serialización y Deserialización
Entrega: Lab 6 - Cliente de Ingesta, Parsing y Resiliencia para APIs de IA
Autor: Emilio Monge (Hidraulia)
=============================================================================
Descripción:
Script integral que implementa un pipeline robusto y resiliente para el
consumo de APIs de IA generativa (Google Gemini / AI Studio), aplicando:
  - Modelos estrictos de validación con Pydantic v2 (tipado fuerte, rangos ge/le).
  - Manejo de excepciones transitorias (TransientAPIError para 429, 5xx y timeouts).
  - Políticas de reintento con Exponential Backoff usando Tenacity.
  - Procesamiento por lotes de reseñas de clientes (prompts de negocio).
  - Serialización estructurada e indentada en 'resultados_ingesta.json'.
  - Modo híbrido: conexión directa con Gemini API (vía GEMINI_API_KEY)
    y simulación determinista para ejecución reproducible sin costes/cuotas.
=============================================================================
"""

import os
import sys
import json
import time
import uuid
import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

import requests
from pydantic import BaseModel, Field, field_validator
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging
from dotenv import load_dotenv

# Carga variables de entorno (.env) si existe
load_dotenv()

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("APIParsingLab6")


# =============================================================================
# 1. EXCEPCIONES Y CONTROL DE ERRORES TRANSITORIOS
# =============================================================================
class TransientAPIError(Exception):
    """
    Excepción personalizada para fallos recuperables o temporales de red y API:
    - HTTP 429: Too Many Requests (Rate limit alcanzado)
    - HTTP 502: Bad Gateway
    - HTTP 503: Service Unavailable
    - HTTP 504: Gateway Timeout
    - Excepciones de Timeout de socket / conexión
    """
    pass


class NonTransientAPIError(Exception):
    """Excepción para fallos no recuperables (HTTP 400 Bad Request, 401 Unauthorized, etc.)."""
    pass


# =============================================================================
# 2. MODELOS PYDANTIC V2 PARA VALIDACIÓN Y PERSISTENCIA
# =============================================================================
class SentimentEnum(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


class AnalisisResena(BaseModel):
    """
    Modelo Pydantic que valida el JSON devuelto por la API del LLM.
    Garantiza que la inferencia cumple exactamente con el esquema de negocio.
    """
    sentiment: SentimentEnum = Field(
        ...,
        description="Clasificación del sentimiento: POSITIVE, NEGATIVE o NEUTRAL",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Nivel de confianza o intensidad emocional (entre 0.0 y 1.0)",
    )
    tema: str = Field(
        ...,
        min_length=2,
        description="Categoría o departamento asignado (ej. Soporte, Facturación)",
    )
    resumen: str = Field(
        ...,
        min_length=5,
        description="Resumen conciso y claro de los puntos clave de la opinión",
    )
    recommendation: str = Field(
        ...,
        min_length=5,
        description="Acción operacional recomendada para el equipo de atención",
    )

    @field_validator("score")
    @classmethod
    def validar_precision(cls, v: float) -> float:
        return round(v, 4)


class MetadatosEjecucion(BaseModel):
    """Metadatos de auditoría y rendimiento técnico del pipeline."""
    timestamp_utc: str
    latencia_ms: float
    proveedor: str
    modo: str


class ExpedienteConsolidado(BaseModel):
    """
    Modelo final estructurado que consolida la consulta original, la respuesta
    validada de la IA y los metadatos de auditoría para archivo o vector DB.
    """
    id_expediente: str
    id_llamada: str
    prompt_original: str
    analisis: AnalisisResena
    metadatos: MetadatosEjecucion


# =============================================================================
# 3. CLIENTE RESILIENTE CON POLÍTICA EXPONENTIAL BACKOFF (TENACITY)
# =============================================================================
@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(TransientAPIError),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def llamar_api_con_resiliencia(
    prompt_texto: str,
    api_key: Optional[str] = None,
    forzar_reintento_demo: bool = False,
) -> Dict[str, Any]:
    """
    Ejecuta la llamada HTTP a la API de IA (o simulación determinista controlada)
    protegida con Tenacity ante errores de concurrencia y límites de tasa.
    """
    t_inicio = time.time()

    # Si forzamos una prueba de reintento para demostrar la captura de TransientAPIError
    if forzar_reintento_demo:
        logger.warning("Simulando HTTP 429 Too Many Requests para validar reintentos de Tenacity...")
        raise TransientAPIError("Simulación de saturación de cuota (HTTP 429).")

    # Caso 1: Se dispone de API Key de Google Gemini
    if api_key and api_key.strip():
        # Endpoint oficial de Google Gemini REST API
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        
        system_instruction = (
            "Eres un analizador semántico de feedback de clientes. "
            "Debes responder EXCLUSIVAMENTE un objeto JSON válido con los siguientes campos: "
            "sentiment (POSITIVE|NEGATIVE|NEUTRAL), score (float entre 0.0 y 1.0), "
            "tema (string), resumen (string), recommendation (string). "
            "No incluyas formato markdown ni bloques ```json."
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"{system_instruction}\n\nReseña a analizar: \"{prompt_texto}\""}
                    ]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.1,
            },
        }

        try:
            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10.0,
            )

            # Clasificación de códigos de estado para resiliencia
            if response.status_code in (429, 502, 503, 504):
                logger.warning(f"Error recuperable HTTP {response.status_code}. Activando reintento...")
                raise TransientAPIError(f"HTTP {response.status_code}: {response.text}")

            if response.status_code != 200:
                raise NonTransientAPIError(f"Error definitivo HTTP {response.status_code}: {response.text}")

            data = response.json()
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
            # Limpieza defensiva en caso de delimitadores markdown
            clean_text = raw_text.strip().removeprefix("```json").removesuffix("```").strip()
            resultado_dict = json.loads(clean_text)
            return {
                "datos": resultado_dict,
                "latencia_ms": round((time.time() - t_inicio) * 1000, 2),
                "proveedor": "Google Gemini (gemini-1.5-flash)",
                "modo": "LIVE_API",
            }

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            logger.warning(f"Fallo de red o timeout ({type(exc).__name__}). Reintentando con Tenacity...")
            raise TransientAPIError(f"Fallo de conexión transitorio: {exc}") from exc

    # Caso 2: Modo estructurado determinista (sin coste / sin cuota)
    else:
        # Simula tiempo de procesamiento de red realista
        time.sleep(0.15)
        simulado = inferir_clasificacion_heuristica(prompt_texto)
        return {
            "datos": simulado,
            "latencia_ms": round((time.time() - t_inicio) * 1000, 2),
            "proveedor": "Simulador Cognitivo Local (compatible con API Gemini)",
            "modo": "DETERMINISTIC_PARSER",
        }


def inferir_clasificacion_heuristica(texto: str) -> Dict[str, Any]:
    """Generador semántico estructurado conforme al esquema requerido por Lab 6."""
    t_lower = texto.lower()

    if any(p in t_lower for p in ["excelente", "rápido", "encantado", "maravilla", "perfecto", "agradecido"]):
        return {
            "sentiment": "POSITIVE",
            "score": 0.94,
            "tema": "Atención al Cliente y Soporte",
            "resumen": "El cliente manifiesta alta satisfacción por la prontitud y la amabilidad del servicio.",
            "recommendation": "Registrar caso de éxito y felicitar al operador de soporte.",
        }
    elif any(p in t_lower for p in ["fallo", "error", "caída", "pésimo", "lento", "bug", "roto", "retraso"]):
        return {
            "sentiment": "NEGATIVE",
            "score": 0.88,
            "tema": "Infraestructura y Estabilidad Técnica",
            "resumen": "Se reportan incidencias recurrentes de conectividad y demora en la entrega.",
            "recommendation": "Abrir ticket de prioridad alta con el equipo de DevOps/Sistemas.",
        }
    else:
        return {
            "sentiment": "NEUTRAL",
            "score": 0.65,
            "tema": "Consultas y Facturación General",
            "resumen": "Petición estándar de información acerca de tarifas y condiciones contractuales.",
            "recommendation": "Enviar folleto explicativo actualizado y programar seguimiento comercial.",
        }


# =============================================================================
# 4. PIPELINE DE INGESTA POR LOTES Y SERIALIZACIÓN
# =============================================================================
def ejecutar_pipeline_ingesta(
    prompts: List[str],
    api_key: Optional[str] = None,
    fichero_salida: str = "resultados_ingesta.json",
) -> List[ExpedienteConsolidado]:
    """
    Procesa un lote de reseñas, valida cada respuesta con Pydantic,
    construye el expediente consolidado y exporta a disco con formato indentado.
    """
    logger.info(f"Iniciando pipeline de ingesta para {len(prompts)} reseñas...")
    expedientes: List[ExpedienteConsolidado] = []

    for idx, prompt in enumerate(prompts, start=1):
        id_call = f"CALL-2026-{idx:04d}"
        logger.info(f"[{idx}/{len(prompts)}] Procesando ({id_call}): \"{prompt[:45]}...\"")

        try:
            # Llamada protegida con reintentos exponenciales
            respuesta_cruda = llamar_api_con_resiliencia(prompt, api_key=api_key)

            # Deserialización y validación estricta con Pydantic
            analisis_validado = AnalisisResena.model_validate(respuesta_cruda["datos"])

            metadatos = MetadatosEjecucion(
                timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                latencia_ms=respuesta_cruda["latencia_ms"],
                proveedor=respuesta_cruda["proveedor"],
                modo=respuesta_cruda["modo"],
            )

            expediente = ExpedienteConsolidado(
                id_expediente=str(uuid.uuid4())[:8],
                id_llamada=id_call,
                prompt_original=prompt,
                analisis=analisis_validado,
                metadatos=metadatos,
            )

            expedientes.append(expediente)
            logger.info(
                f"    -> Sentimiento: {analisis_validado.sentiment.value} | "
                f"Score: {analisis_validado.score:.2f} | Latencia: {metadatos.latencia_ms}ms"
            )

        except Exception as err:
            logger.error(f"Fallo irrecuperable en elemento {id_call}: {err}")

    # Serialización y exportación a JSON con indent=2 (Paso 7)
    logger.info(f"Guardando resultados validados en '{fichero_salida}'...")
    datos_exportables = [exp.model_dump(mode="json") for exp in expedientes]

    with open(fichero_salida, "w", encoding="utf-8") as f:
        json.dump(datos_exportables, f, ensure_ascii=False, indent=2)

    logger.info(f"Pipeline finalizado con éxito. {len(expedientes)} registros guardados en '{fichero_salida}'.")
    return expedientes


# =============================================================================
# 5. DEMOSTRACIÓN DE RESILIENCIA Y EJECUCIÓN PRINCIPAL
# =============================================================================
def test_resiliencia_tenacity():
    """Valida que el decorador @retry captura TransientAPIError y aplica backoff."""
    logger.info("--- TEST UNITARIO DE RESILIENCIA CON TENACITY ---")
    intentos = 0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.2, min=0.2, max=1.0),
        retry=retry_if_exception_type(TransientAPIError),
        reraise=True,
    )
    def funcion_inestable():
        nonlocal intentos
        intentos += 1
        if intentos < 3:
            logger.info(f"Intento {intentos}: Simulando error temporal HTTP 429...")
            raise TransientAPIError("Límite de tasa alcanzado")
        logger.info(f"Intento {intentos}: Conexión recuperada con éxito!")
        return "OK_RECUPERADO"

    resultado = funcion_inestable()
    assert resultado == "OK_RECUPERADO"
    assert intentos == 3
    logger.info("Test de resiliencia completado satisfactoriamente.\n")


def main():
    print("=" * 75)
    print(" PROGRAMACIÓN DE IA - LABORATORIO 6: INGESTA, PARSING Y RESILIENCIA ")
    print("=" * 75)

    # 1. Ejecutar test de resiliencia de Tenacity
    test_resiliencia_tenacity()

    # 2. Obtener clave de API si está disponible en entorno
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        logger.info("Clave GEMINI_API_KEY detectada. Se realizarán llamadas reales a Google AI Studio.")
    else:
        logger.info("No se encontró GEMINI_API_KEY. Se utilizará el motor de parsing semántico local.")
        logger.info("Para usar Gemini real, añade tu clave en un archivo .env o variable GEMINI_API_KEY.")

    # 3. Lote de 5 prompts de negocio (reseñas de clientes con distintas casuísticas)
    prompts_negocio = [
        "El servicio de atención telefónica fue impecable, resolvieron mi problema con la factura en menos de cinco minutos.",
        "La aplicación lleva caída toda la mañana con error 500 y me ha bloqueado la entrega a clientes. Es inadmisible.",
        "Quisiera solicitar información detallada sobre las condiciones del plan anual y las formas de pago disponibles.",
        "Los envíos siempre llegan con un día de retraso y los paquetes vienen con marcas de golpes. Deben mejorar la logística.",
        "Una maravilla de plataforma, la interfaz es intuitiva y el soporte técnico nos atendió de diez.",
    ]

    # 4. Ejecución del pipeline
    fichero_destino = "resultados_ingesta.json"
    resultados = ejecutar_pipeline_ingesta(prompts_negocio, api_key=api_key, fichero_salida=fichero_destino)

    # 5. Muestra del resultado serializado
    print("\n" + "=" * 75)
    print(" MUESTRA DEL PRIMER REGISTRO DESERIALIZADO Y VALIDADO (Pydantic model_dump_json):")
    print("=" * 75)
    if resultados:
        print(resultados[0].model_dump_json(indent=2))

    print(f"\n[OK] Ejercicio completado. Revisa el archivo generado: {fichero_destino}\n")


if __name__ == "__main__":
    main()
