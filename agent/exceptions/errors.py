class AgentError(Exception):
    """Base exception for all Agent errors."""
    pass

class PlannerError(AgentError):
    """Raised when the Planner fails to generate a valid plan."""
    pass

class CoderError(AgentError):
    """Raised when the Coder fails to generate a valid patch."""
    pass

class ExecutionError(AgentError):
    """Raised when a command execution fails or times out."""
    pass

class FileOperationError(AgentError):
    """Raised when a file operation (read/write/delete/create) fails or violates security."""
    pass

class LLMError(AgentError):
    """Raised when the LLM client encounters an error."""
    pass

class PreflightError(AgentError):
    """Raised when pre-run validation of a role's provider/model/credentials fails."""
    pass
