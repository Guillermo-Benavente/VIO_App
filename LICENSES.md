# Política de licencias de modelos

VIO no descarga modelos automáticamente. Antes de distribuir un producto, conserva una copia de la licencia y la *model card* exactas de cada peso que empaquetes.

Los adaptadores incluidos están pensados para estas opciones permisivas:

| Componente | Opción | Licencia a verificar |
|---|---|---|
| STT | OpenAI Whisper o conversiones `faster-whisper` de Systran | MIT |
| LLM | Un GGUF basado en un modelo Apache-2.0 | Apache-2.0 |
| TTS/zero-shot | Chatterbox Multilingual y sus pesos oficiales | MIT |
| Motor LLM | llama.cpp / llama-cpp-python | MIT |

No uses en el perfil distribuido un modelo con licencia de investigación, "non-commercial", Llama Community License, ni términos de proveedor que no aceptes. La licencia del código y la de los pesos pueden ser distintas: la model card del archivo GGUF concreto manda.

La clonación de voz requiere autorización de la persona cuya voz se use. VIO no incluye voces ni archivos de referencia.
