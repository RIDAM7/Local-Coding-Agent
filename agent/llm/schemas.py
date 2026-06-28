from typing import Optional, Dict, Any
from pydantic import BaseModel

class OllamaRequest(BaseModel):
    model: str
    prompt: str
    format: Optional[str] = None
    stream: bool = False
    options: Optional[Dict[str, Any]] = None

class OllamaResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool
    context: Optional[list[int]] = None
    total_duration: Optional[int] = None
