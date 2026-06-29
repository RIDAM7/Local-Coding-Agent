"""Schema for the optional Phase 3 prompt refiner stage."""

from typing import List

from pydantic import BaseModel, Field


class RefinedPrompt(BaseModel):
    """A structured, clarified rewrite of a raw user task.

    The refiner only disambiguates and structures the original request. It must
    NOT invent new scope, features, or requirements — every field is derived
    strictly from what the user already asked for.
    """

    refined_task: str = Field(
        ...,
        description="The rewritten task instruction: clearer and more complete, "
        "but preserving the original intent and constraints. No new scope.",
    )
    clarified_goal: str = Field(
        ...,
        description="A one-line statement of the user's underlying goal.",
    )
    assumptions: List[str] = Field(
        default_factory=list,
        description="Implicit assumptions made explicit. Must not add new requirements.",
    )
    acceptance_criteria: List[str] = Field(
        default_factory=list,
        description="Concrete, checkable conditions that define 'done', derived "
        "only from the original request.",
    )
    open_questions: List[str] = Field(
        default_factory=list,
        description="Genuine ambiguities the user may want to resolve. Optional.",
    )
