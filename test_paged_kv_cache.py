
from paged_kv_cache import PagedKVCache


def main():
    cache = PagedKVCache(total_blocks=2, block_size=2)

    # Test 1: Fill the entire pool with request A.
    cache.create_request("A")

    for _ in range(4):
        cache.append_token("A")

    assert len(cache.free_blocks) == 0
    assert len(cache.sequences["A"].block_ids) == 2

    print("PASS: Request A consumed both blocks.")

    # Test 2: B cannot allocate when the pool is exhausted.
    cache.create_request("B")

    try:
        cache.append_token("B")
        raise AssertionError("Expected MemoryError was not raised")
    except MemoryError:
        print("PASS: Allocation fails safely when the pool is full.")

    # Failed allocation must not corrupt B's state.
    assert cache.sequences["B"].token_count == 0
    assert cache.sequences["B"].block_ids == []

    # Test 3: Releasing A should return its blocks.
    cache.release_request("A")

    assert len(cache.free_blocks) == 2
    assert "A" not in cache.sequences

    print("PASS: Releasing A returned both blocks.")

    # Test 4: B can now allocate and append tokens.
    cache.append_token("B")
    cache.append_token("B")
    cache.append_token("B")

    assert cache.sequences["B"].token_count == 3
    assert len(cache.sequences["B"].block_ids) == 2
    assert len(cache.free_blocks) == 0

    print("PASS: Request B reused the available capacity.")

    # Test 5: Verify logical-to-physical mapping.
    for position in range(3):
        block, offset = cache.physical_location("B", position)
        print(
            f"B token {position} -> "
            f"physical block {block}, offset {offset}"
        )

    print("\nAll allocator tests passed.")
    cache.report()


if __name__ == "__main__":
    main()
