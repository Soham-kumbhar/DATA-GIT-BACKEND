from pydantic import BaseModel, Field


class AIQuestionRequest(BaseModel):
    version_1: int | None = None
    version_2: int | None = None

    question: str = Field(
        min_length=1,
        max_length=2000,
    )


class AIQuestionResponse(BaseModel):
    status: str
    question: str
    answer: str

    evidence: list[str] = Field(
        default_factory=list
    )

    limitations: list[str] = Field(
        default_factory=list
    )

    confidence: str


class AIImpactAssessment(BaseModel):
    area: str
    status: str
    explanation: str

    evidence: list[str] = Field(
        default_factory=list
    )