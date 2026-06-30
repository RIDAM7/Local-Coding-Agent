import os

class BudgetManager:
    def __init__(self, enforcement_enabled: bool = True):
        self.enforcement_enabled = enforcement_enabled
        self.session_cost = 0.0
        self.max_session_cost = float(os.environ.get("MAX_SESSION_COST_USD", "2.0"))
        
    def add_cost(self, cost: float):
        self.session_cost += cost
        
    def can_afford(self) -> bool:
        if not self.enforcement_enabled:
            return True
        return self.session_cost < self.max_session_cost
        
    def check_payload(self, estimated_tokens: int) -> bool:
        if not self.enforcement_enabled:
            return True
        return estimated_tokens <= 60000
