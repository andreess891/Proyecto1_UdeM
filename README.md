# Clasificación ordenes de desviación de consumos

**Asignatura:** Proyecto 2, Especialización en ciencia de datos e inteligencia artificial
**Universidad:** Universidad de Medellín

---

## Descripción de la Iniciativa

### Problema/Necesidad u Oportunidad Identificada

De conformidad con el artículo 149 de la Ley 142, toda empresa de servicios públicos domiciliarios está obligada a investigar las desviaciones significativas en los consumos registrados.

En el caso de EPM, ante cada situación de desviación, se genera una orden que debe ser gestionada por un analista, quien realiza la correspondiente investigación. Actualmente, este proceso se realiza de manera completamente manual, involucrando a numerosos analistas en la revisión de órdenes, lo que incrementa el riesgo de errores humanos.

El propósito de la iniciativa es optimizar el tiempo de dedicación de los analistas a esta labor, que representa aproximadamente el **90% de su ocupación diaria**. Esto permite liberar recursos que puedan ser destinados a actividades de mayor valor para la organización.

### Escenarios Actuales

Para la legalización de estas órdenes, pueden presentarse los siguientes escenarios:

* El analista decide modificar el consumo.
* El analista decide modificar el consumo y remitir el caso a una cuadrilla para efectuar una investigación en terreno.
* El analista opta por no modificar el consumo y remitir el caso a una cuadrilla para una investigación en terreno.
* El analista decide no modificar el consumo.

### Efectos del Problema

* Riesgos en dejar pasar consumos erróneos al cliente.
* Incremento en los reclamos del cliente.

---

## Objetivo del Proyecto

Desarrollar un modelo supervisado de clasificación capaz de analizar las órdenes previamente revisadas por los analistas, aprendiendo de sus decisiones, con el fin de aplicar este aprendizaje al procesamiento automático de nuevas desviaciones detectadas en los consumos.

---

## Alternativa de Solución

### Contexto (Proyecto 1)

El alcance de Proyecto 1 consistió en evaluar dos modelos de clasificación, **Random Forest** y **XGBoost**, utilizando diversas métricas. No obstante, todo este proceso se llevó a cabo en un notebook y sin realizar ajuste de hiperparámetros ni mejores prácticas de modelado.

### Alcance (Proyecto 2)

El propósito de Proyecto 2 es implementar una estrategia que no solo permita evaluar estos modelos, sino también incorporar otros adicionales, aplicando **mejores prácticas de MLOps**.

Esto abarca:

1. El seguimiento y registro de los modelos para su adecuada comparación.
2. La orquestación y el despliegue del modelo que demuestre el mejor desempeño.

### Descripción de la Alternativa

1. **Estructuración**: Iniciaremos estructurando la solución, asegurándonos de que se adapte correctamente a las particularidades y metodologías propias del entorno donde se desarrollará el proyecto.
2. **Experimentación**: Procederemos con la fase de experimentación, en la que probaremos distintos modelos y técnicas. Durante este proceso, utilizaremos **MLFlow** para registrar y hacer seguimiento detallado de cada modelo, aprovechando una arquitectura modular definida previamente.
3. **Orquestación y Despliegue**: Finalmente, implementaremos la orquestación del flujo de trabajo mediante **PREFECT**, abarcando desde la ingesta de datos hasta el despliegue del modelo que presente las mejores métricas de desempeño.
