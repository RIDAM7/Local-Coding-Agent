import asyncio
from agent.llm.client import OllamaClient
from agent.repair.constraints import ConstraintExtractor
from agent.models.schemas import Constraint, Patch, FileOperation, RepairScope
from agent.repair.constraint_validator import ConstraintValidator
from agent.repair.rollback import RollbackManager
import json

async def main():
    print("=== Constraint Extraction Audit ===")
    client = OllamaClient()
    extractor = ConstraintExtractor(client)
    
    tasks = [
        "Do NOT modify tests.",
        "Only modify calculator.py.",
        "Do not delete files.",
        "Leave existing tests untouched.",
        "Only modify files under src/auth."
    ]
    
    for task in tasks:
        print(f"\nTask: {task}")
        res = await extractor.extract(task)
        print(f"Success: {res.success}")
        for c in res.constraints:
            print(f"- {c.model_dump_json()}")

    print("\n\n=== Repair Scope Audit ===")
    
    # Scenario 1
    print("\nScenario 1:")
    initial_patch = Patch(operations=[FileOperation(type="update_file", path="calculator.py", content="")])
    scope = RepairScope(allowed_paths=["calculator.py"])
    repair_patch = Patch(operations=[FileOperation(type="update_file", path="test_calculator.py", content="")])
    print(f"Original patch targets: {[op.path for op in initial_patch.operations]}")
    print(f"Generated RepairScope: {scope.allowed_paths}")
    print(f"Repair patch path: {[op.path for op in repair_patch.operations]}")
    res = ConstraintValidator.validate(repair_patch, constraints=[], repair_scope=scope)
    print(f"Acceptance/Rejection: {'ACCEPTED' if res.is_valid else 'REJECTED: ' + str(res.violations)}")

    # Scenario 2
    print("\nScenario 2:")
    initial_patch = Patch(operations=[FileOperation(type="create_file", path="new_module.py", content="")])
    scope = RepairScope(allowed_paths=["new_module.py"])
    repair_patch = Patch(operations=[FileOperation(type="update_file", path="new_module.py", content="updated")])
    print(f"Original patch targets: {[op.path for op in initial_patch.operations]}")
    print(f"Generated RepairScope: {scope.allowed_paths}")
    print(f"Repair patch path: {[op.path for op in repair_patch.operations]}")
    res = ConstraintValidator.validate(repair_patch, constraints=[], repair_scope=scope)
    print(f"Acceptance/Rejection: {'ACCEPTED' if res.is_valid else 'REJECTED: ' + str(res.violations)}")
    
    # Scenario 3
    print("\nScenario 3:")
    initial_patch = Patch(operations=[FileOperation(type="update_file", path="src/main.py", content="")])
    scope = RepairScope(allowed_paths=["src/main.py"])
    repair_patch = Patch(operations=[
        FileOperation(type="update_file", path="src/main.py", content=""),
        FileOperation(type="update_file", path="src/utils.py", content="")
    ])
    print(f"Original patch targets: {[op.path for op in initial_patch.operations]}")
    print(f"Generated RepairScope: {scope.allowed_paths}")
    print(f"Repair patch path: {[op.path for op in repair_patch.operations]}")
    res = ConstraintValidator.validate(repair_patch, constraints=[], repair_scope=scope)
    print(f"Acceptance/Rejection: {'ACCEPTED' if res.is_valid else 'REJECTED: ' + str(res.violations)}")

    print("\n\n=== Rollback Verification Audit ===")
    from agent.config import settings
    workspace = settings.get_workspace_path()
    
    # Setup dummy file
    dummy_file = workspace / "dummy_rollback_test.py"
    dummy_file.write_text("print('baseline')", encoding="utf-8")
    
    rm = RollbackManager()
    rm.checkpoint(["dummy_rollback_test.py"])
    
    print("Stored hashes:")
    for path, (content, hash_val) in rm.file_backups.items():
        print(f"{path}: {hash_val}")
        
    # Simulate a bad repair
    dummy_file.write_text("print('bad repair')", encoding="utf-8")
    
    # Restore
    rm.restore()
    
    print("\nRestored hashes (verification result):")
    results = rm.verify()
    for path, success in results.items():
        print(f"{path} -> {'SUCCESS (hash matched baseline)' if success else 'FAILURE (hash mismatch)'}")
        
    dummy_file.unlink()

if __name__ == "__main__":
    asyncio.run(main())
