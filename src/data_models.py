from pydantic import BaseModel, Field
from typing import Dict, Optional, Any, List, Union
from uuid import UUID


class SurveyResponse(BaseModel):
    """Model representing a response option with its count."""
    option: str
    count: Optional[int] = None

    @classmethod
    def from_dict(cls, response_dict: Dict[str, Any]) -> List["SurveyResponse"]:
        """Convert a response dictionary to a list of SurveyResponse objects."""
        responses = []
        for option, count in response_dict.items():
            # Handle empty count values
            if count == "":
                count = None
            responses.append(cls(option=option, count=count))
        return responses


class SurveyQuestion(BaseModel):
    """Model representing a survey question with its metadata and responses."""
    id: UUID
    variable_name: str = Field(..., alias="variableName")
    description: str
    section: str = Field(..., alias="Section")
    level: str = Field(..., alias="Level")
    type: str = Field(..., alias="Type")
    width: str = Field(..., alias="Width")
    decimals: str = Field(..., alias="Decimals")
    cai_reference: str = Field("", alias="CAI Reference")
    question: str
    response: Dict[str, Any]  # Raw response data
    wave: str

    # Processed responses
    response_items: List[SurveyResponse] = []

    class Config:
        populate_by_name = True

    def __init__(self, **data):
        super().__init__(**data)
        # Process response items after initialization
        self.response_items = SurveyResponse.from_dict(self.response)


class SurveyData(BaseModel):
    """Container for multiple survey questions."""
    questions: List[SurveyQuestion] = []

    @classmethod
    def from_json(cls, data: List[Dict[str, Any]]) -> "SurveyData":
        """Create a SurveyData object from JSON data."""
        return cls(questions=[SurveyQuestion(**item) for item in data])


class EmbeddingRecord(BaseModel):
    """Model for storing document embeddings in ChromaDB."""
    id: str
    question_id: str
    variable_name: str
    description: str
    question_text: str
    response_summary: str
    wave: str
    section: str
    level: str
    metadata: Dict[str, Any] = {}
    embedding: Optional[List[float]] = None


class QueryResult(BaseModel):
    """Model for query results returned to the frontend."""
    question: SurveyQuestion
    similarity_score: float = 0.0
    relevance_explanation: str = ""


class UserQuery(BaseModel):
    """Model for incoming user queries."""
    query_text: str
    filters: Dict[str, Any] = {}
    limit: int = 20