
from dataclasses import dataclass, field


@dataclass
class SequenceState:
    request_id: str
    token_count: int = 0
    block_ids: list[int] = field(default_factory=list)


class PagedKVCache:
    def __init__(self, total_blocks: int, block_size: int):
        if total_blocks <= 0 or block_size <= 0:
            raise ValueError("total_blocks and block_size must be positive")

        self.total_blocks = total_blocks
        self.block_size = block_size

        # Initially, every physical block is free.
        self.free_blocks = list(range(total_blocks))

        # Maps request IDs to their sequence state.
        self.sequences: dict[str, SequenceState] = {}

    def create_request(self, request_id: str):
        if request_id in self.sequences:
            raise ValueError(f"Request already exists: {request_id}")

        self.sequences[request_id] = SequenceState(
            request_id=request_id
        )

    def append_token(self, request_id: str):
        if request_id not in self.sequences:
            raise KeyError(f"Unknown request: {request_id}")

        sequence = self.sequences[request_id]

        # Allocate a new physical block when the current one is full.
        if sequence.token_count % self.block_size == 0:
            if not self.free_blocks:
                raise MemoryError("KV-cache block pool exhausted")

            block_id = self.free_blocks.pop()
            sequence.block_ids.append(block_id)

        sequence.token_count += 1

    def physical_location(self, request_id: str, token_position: int):
        sequence = self.sequences[request_id]

        if not 0 <= token_position < sequence.token_count:
            raise IndexError("Token position is outside the sequence")

        logical_block = token_position // self.block_size
        offset_in_block = token_position % self.block_size

        physical_block = sequence.block_ids[logical_block]

        return physical_block, offset_in_block

    def release_request(self, request_id: str):
        sequence = self.sequences.pop(request_id)

        # Return all of this request's physical blocks to the free pool.
        self.free_blocks.extend(sequence.block_ids)

    def report(self):
        used_blocks = self.total_blocks - len(self.free_blocks)
        used_slots = sum(
            sequence.token_count for sequence in self.sequences.values()
        )
        capacity_slots = used_blocks * self.block_size

        print(f"Total blocks: {self.total_blocks}")
        print(f"Block size: {self.block_size} tokens")
        print(f"Used blocks: {used_blocks}")
        print(f"Free blocks: {len(self.free_blocks)}")
        print(f"Active requests: {len(self.sequences)}")
        print(f"Stored tokens: {used_slots}")
        print(f"Allocated token slots: {capacity_slots}")
        print(f"Unused slots in allocated blocks: {capacity_slots - used_slots}")


def main():
    cache = PagedKVCache(total_blocks=8, block_size=4)

    cache.create_request("request-A")
    cache.create_request("request-B")

    for _ in range(6):
        cache.append_token("request-A")

    for _ in range(3):
        cache.append_token("request-B")

    print("=== Cache after generating tokens ===")
    cache.report()

    print("\n=== Request A logical-to-physical mapping ===")
    for position in range(6):
        block, offset = cache.physical_location("request-A", position)
        print(
            f"Token {position}: physical block {block}, "
            f"offset {offset}"
        )

    print("\n=== Release request A ===")
    cache.release_request("request-A")
    cache.report()


if __name__ == "__main__":
    main()
