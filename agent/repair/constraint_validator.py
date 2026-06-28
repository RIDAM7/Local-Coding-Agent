import fnmatch
from agent.models.schemas import Patch, Constraint, RepairScope, ConstraintValidationResult

class ConstraintValidator:
    @staticmethod
    def validate(patch: Patch, constraints: list[Constraint], repair_scope: RepairScope | None = None) -> ConstraintValidationResult:
        violations = []
        
        for op in patch.operations:
            target_path = op.path.replace('\\', '/')
            
            # Repair Scope lock
            if repair_scope:
                if target_path not in repair_scope.allowed_paths:
                    violations.append(f"Attempted modification outside repair scope: {target_path}")
            
            # Constraints
            for c in constraints:
                if c.type == "PROTECTED_PATH" and c.patterns:
                    for pattern in c.patterns:
                        if fnmatch.fnmatch(target_path, pattern) or fnmatch.fnmatch(target_path.split('/')[-1], pattern):
                            violations.append(f"Attempted modification of protected file: {target_path}")
                            break
                            
                elif c.type == "ALLOWLIST_PATH" and c.patterns:
                    allowed = False
                    for pattern in c.patterns:
                        if fnmatch.fnmatch(target_path, pattern) or fnmatch.fnmatch(target_path.split('/')[-1], pattern):
                            allowed = True
                            break
                    if not allowed:
                        violations.append(f"Target '{target_path}' is not in the allowlist.")
                        
                elif c.type == "NO_DELETE":
                    if op.type == "delete_file":
                        violations.append(f"Attempted to delete file '{target_path}', but NO_DELETE constraint is active.")
                        
        return ConstraintValidationResult(
            is_valid=len(violations) == 0,
            violations=violations
        )
