import asyncio
import os
from pathlib import Path
from agent.memory.manager import MemoryManager
from agent.memory.schemas import MemoryRecord, MemoryType, MemoryMetadata

async def audit():
    print("--- Memory Foundation Adversarial Audit ---")
    
    mem_dir = Path("~/.antigravity_data/memory").expanduser()
    mem = MemoryManager(mem_dir)
    if not mem.enabled:
        print("FAIL: MemoryManager disabled.")
        return
        
    print("\n[1] Duplicate Memory Detection")
    # Store a memory
    rec1 = MemoryRecord(
        memory_id="test_dup_1",
        memory_type=MemoryType.ORACLE_SOLUTION,
        importance_score=1.0,
        embedding_text="Duplicate test oracle",
        content="This is the first insertion.",
        metadata=MemoryMetadata(task="dup_test", source_id="test_dup", workspace_fingerprint=["python"])
    )
    
    id1 = await mem.store(rec1)
    
    # Store it again
    rec2 = MemoryRecord(
        memory_id="test_dup_1",
        memory_type=MemoryType.ORACLE_SOLUTION,
        importance_score=1.0,
        embedding_text="Duplicate test oracle",
        content="This is the second insertion.",
        metadata=MemoryMetadata(task="dup_test", source_id="test_dup", workspace_fingerprint=["python"])
    )
    
    id2 = await mem.store(rec2)
    
    # Fetch collection size
    results = mem.collection.get(ids=["test_dup_1"])
    if len(results['ids']) == 1:
        print("PASS: Duplicate memory IDs are overwritten, preventing inflation.")
    else:
        print(f"FAIL: Duplicate IDs inserted. Size: {len(results['ids'])}")

    print("\n[2] Retrieval Quality (Top-K Dominance)")
    recs = await mem.search("Duplicate test oracle", ["python"], limit=5)
    print(f"Returned {len(recs)} records for query.")
    for r in recs:
        print(f" - {r.memory_id} (Score: {r.final_score:.2f})")
        
    ids = [r.memory_id for r in recs]
    if len(ids) == len(set(ids)) and "test_dup_1" in ids:
        print("PASS: Top-k retrieves distinct memories without duplicate spam.")
    else:
        print("FAIL: Duplicates found in top-k or target missing.")
        
    print("\n[3] Constraint Safety & Scrubbing")
    # Path scrubbing
    scrubbed = mem._scrub_paths("Error in C:\\Users\\Name\\project\\main.py")
    if "[SCRUBBED_PATH]" in scrubbed and "C:\\Users" not in scrubbed:
        print("PASS: Path scrubbing strips absolute paths.")
    else:
        print(f"FAIL: Scrubbing ineffective: {scrubbed}")
        
    # Constraint isolation
    rec_java = MemoryRecord(
        memory_id="test_java_1",
        memory_type=MemoryType.ORACLE_SOLUTION,
        importance_score=1.0,
        embedding_text="Java oracle fix",
        content="Java solution.",
        metadata=MemoryMetadata(task="java_test", source_id="java_test", workspace_fingerprint=["java"])
    )
    await mem.store(rec_java)
    
    java_res = await mem.search("Java oracle fix", ["python"], limit=1)
    if not java_res or java_res[0].memory_id != "test_java_1":
         print("PASS: Fingerprint mismatch successfully prevented retrieval.")
    else:
         print("FAIL: Fingerprint isolation bypassed.")

if __name__ == "__main__":
    asyncio.run(audit())
