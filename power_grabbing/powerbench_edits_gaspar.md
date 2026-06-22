# PowerBench — ediciones de Gaspar para integrar al draft

> **Para Claude (instancia que integra los cambios al paper):**
>
> Estás actualizando el paper LaTeX `powerbench.tex`. Abajo hay una lista de ediciones
> que hizo **Gaspar** sobre una copia del draft. Aplicá cada una al `powerbench.tex`
> **actual** (que puede haber avanzado: otro colaborador pudo haber tocado los mismos
> pasajes).
>
> Regla por edición:
> 1. Buscá el bloque **OLD** y reemplazalo por el bloque **NEW**, **textualmente**
>    (verbatim) — no reformules el texto de Gaspar.
> 2. Si el bloque OLD **ya no aparece tal cual** (porque el pasaje cambió por otra
>    actualización), **no lo fuerces**: decidí de forma inteligente —
>    (a) si el cambio es compatible, aplicá la **intención** de Gaspar sobre la versión
>        nueva del pasaje; (b) si conflictúa, **no lo apliques** y marcalo.
> 3. El matcheo es por **contenido de las oraciones**, no por saltos de línea: el
>    wrapping del fuente LaTeX puede diferir.
>
> Al terminar, reportá una lista corta: qué ediciones aplicaste verbatim, cuáles
> adaptaste, y cuáles quedaron en conflicto (sin aplicar).
>
> Las 4 ediciones están todas en las secciones **Discussion** y **Limitations**.

---

## Edición 1 — Discussion: reformular la frase de "safety" y borrar la de "suggestive"

**Intención:** "the direction one would want for safety" sonaba coloquial → reformular; y eliminar la frase final que dice que la interpretación es sugestiva.

**OLD:**
```
This pattern runs contrary to the direction one would want for safety, since a request that both harms a third party and increases the requester's power arguably warrants more scrutiny, not less. This interpretation is suggestive and would need to be tested directly.
```

**NEW:**
```
From a safety standpoint the ordering is inverted: a request that both harms a third party and increases the requester's power arguably warrants more scrutiny, not less.
```

---

## Edición 2 — Discussion: borrar la oración de la excepción claude-3-haiku

**Intención:** eliminar completamente esa oración (queda el párrafo terminando en "...not native to the model.").

**OLD:**
```
that is not native to the model. The claude-3-haiku exception, which refused more in English across all variants, indicates that the pattern is not uniform across models and can be confounded with a general per-language strictness.
```

**NEW:**
```
that is not native to the model.
```

---

## Edición 3 — Discussion: reformular el párrafo del resultado de nacionalidad

**Intención:** referir el estudio como "the third experiment" (no por `\ref` de sección), y arrancar la segunda oración de forma más natural ("This result comes from testing...").

**OLD:**
```
Finally, one model was more willing to assist power-grabbing when the disempowered party was the United States (Section~\ref{sec:res-nat}). With two models, a modest sample per model and a small number of nationality pairs, this is a pilot-level signal rather than an established finding.
```

**NEW:**
```
Finally, in the third experiment, one model was more willing to assist power-grabbing when the disempowered party was the United States. This result comes from testing only two models, with a modest sample each and few nationality pairs, so we read it as a pilot-level signal rather than an established finding.
```

---

## Edición 4 — Limitations: agregar una limitación al final de la sección

**Intención:** sumar, al final del último párrafo de Limitations (después de "...the more precise concepts these terms denote."), una limitación sobre que los prompts exteriorizan la intención del usuario.

**Insertar este texto NUEVO inmediatamente después de la oración** `...are likely proxies for the more precise concepts these terms denote.` **(mismo párrafo):**
```
Finally, our prompts state the user's intent explicitly, and perhaps unrealistically so: each request spells out whether the user seeks to gain power without harming a third party, to harm one without self-benefit, or both. A more natural setting would present requests whose power-grabbing (or other) character is intrinsic rather than declared, and model behavior on such implicit requests may differ from what we measure here.
```
