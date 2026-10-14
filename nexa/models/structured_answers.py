"""Serialize an entity list into a caller's Pydantic answer schema.

Used by the Direct pipeline's length-overflow fallback, which recovers entity
names from the KG result when the LLM runs out of output tokens.
"""
from typing import get_args, get_origin

from pydantic import ValidationError


def format_entity_answer(schema, names, explanation):
    """Return a validated ``schema`` instance carrying ``names``, or None.

    The ``answer`` field may be ``str`` (comma-joined names) or ``list[str]``.
    None means the schema is not an entity list (boolean, count, or other
    required fields), and the caller should give up rather than invent values.
    """
    fields = getattr(schema, 'model_fields', {})
    field = fields.get('answer')
    if field is None or not all(isinstance(name, str) for name in names):
        return None
    annotation = field.annotation
    if annotation is str:
        answer = ','.join(names)
    elif get_origin(annotation) is list and get_args(annotation) == (str,):
        answer = list(names)
    else:
        return None
    payload = {'answer': answer}
    if 'thinking' in fields and fields['thinking'].annotation is str:
        payload['thinking'] = explanation
    try:
        return schema.model_validate(payload)
    except ValidationError:
        return None
