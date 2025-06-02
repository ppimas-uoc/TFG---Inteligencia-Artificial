# TFG – Inteligencia Artificial

*Junio de 2025*

![Logo UOC](https://www.uoc.edu/portal/_resources/common/imatges/marca_UOC/UOC_Masterbrand.jpg)

---

## Aplicación de técnicas de IA fiable en la predicción del índice de calidad de vida en personas con tratamiento oncológico mediante aprendizaje automático

---

**Pablo Pimàs Verge**  
*Grado en Ingeniería Informática*  
*Inteligencia Artificial*

**Dra. María Moreno de Castro**  
**Dr. Friman Sanchéz**

## Tabla de contenidos

1. [Título](#tfg--inteligencia-artificial)
2. [Descripción](#descripción)
4. [Estructura del repositorio](#estructura-del-repositorio)

---

## Descripción

**Aplicación de técnicas de IA fiable en la predicción del índice de calidad de vida en personas con tratamiento oncológico mediante aprendizaje automático.**

En este Trabajo de Fin de Grado (TFG) se aborda el desafío de estimar de forma fiable el Índice de Calidad de Vida (QoL) en pacientes oncológicos, utilizando técnicas de Machine Learning. El propósito principal es construir un modelo que:
- Reciba datos clínicos recopilados con los PROM EORTC QLQ-C30 y EORTC QLQ-C23 de las y los pacientes.
- Prediga el QoL como mejorable o aceptable con una incertidumbre cuantificada.
- Ofrezca interpretabilidad y validación estadística para su posible uso en entornos sanitarios.

---

## Estructura del repositorio

```text
├── carbon/                     # Consumo de energía y emisiones de CO2 de los notebooks
├── data/                       # Conjunto de datos (en crudo)
├── documentation/              # Documentación del proyecto
├── Fase 2/                     # Análisis exploratorio, modelado y XAI
├── Fase 3/                     # Cuantificación de la incertidumbre
├── html notebooks versions/    # Versiones html estáticas de los noteebooks
├── xaiuq_functions             # Paquete de funciones auxiliares