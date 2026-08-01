from .finance import attach_finance
from .loader import (
    filter_segment,
    is_nemotron_persona,
    load_personas,
    persona_source_counts,
    require_nemotron_personas,
    sample_cohort,
    synthesize,
)
from .schema import AGE_BANDS, FinanceProfile, Persona, Segment, age_band

__all__ = [
    "AGE_BANDS",
    "FinanceProfile",
    "Persona",
    "Segment",
    "age_band",
    "attach_finance",
    "filter_segment",
    "is_nemotron_persona",
    "load_personas",
    "persona_source_counts",
    "require_nemotron_personas",
    "sample_cohort",
    "synthesize",
]
