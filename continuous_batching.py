
from dataclasses import dataclass


@dataclass
class Request:
    request_id: str
    arrival_time: int
    max_new_tokens: int

    generated_tokens: int = 0
    start_time: int | None = None
    completion_time: int | None = None

    @property
    def finished(self):
        return self.generated_tokens >= self.max_new_tokens


class ContinuousBatchScheduler:
    def __init__(self, max_active_requests: int):
        if max_active_requests <= 0:
            raise ValueError("max_active_requests must be positive")

        self.max_active_requests = max_active_requests
        self.pending = []
        self.active = []
        self.completed = []

        self.time = 0
        self.busy_slots = 0
        self.total_slot_capacity = 0

    def submit(self, request: Request):
        self.pending.append(request)
        self.pending.sort(key=lambda r: r.arrival_time)

    def admit_requests(self):
        """Fill available slots with requests that have arrived."""
        while (
            self.pending
            and len(self.active) < self.max_active_requests
            and self.pending[0].arrival_time <= self.time
        ):
            request = self.pending.pop(0)
            request.start_time = self.time
            self.active.append(request)

            print(
                f"t={self.time}: admitted {request.request_id} "
                f"(target={request.max_new_tokens} tokens)"
            )

    def generation_step(self):
        """Simulate one decode step for every active request."""
        active_count = len(self.active)

        self.busy_slots += active_count
        self.total_slot_capacity += self.max_active_requests

        for request in list(self.active):
            request.generated_tokens += 1

            print(
                f"t={self.time}: {request.request_id} generated "
                f"token {request.generated_tokens}/"
                f"{request.max_new_tokens}"
            )

            if request.finished:
                request.completion_time = self.time + 1
                self.active.remove(request)
                self.completed.append(request)

                print(
                    f"t={self.time + 1}: {request.request_id} completed"
                )

    def run(self):
        while self.pending or self.active:
            # Admit requests that have arrived by the current time.
            self.admit_requests()

            # Advance time to the next arrival if no request can run.
            if not self.active:
                if self.pending:
                    self.time = max(
                        self.time,
                        self.pending[0].arrival_time,
                    )
                    continue
                break

            self.generation_step()
            self.time += 1

        self.print_report()

    def print_report(self):
        print("\n" + "=" * 60)
        print("CONTINUOUS BATCHING REPORT")
        print("=" * 60)

        for request in sorted(
            self.completed, key=lambda r: r.request_id
        ):
            waiting_time = request.start_time - request.arrival_time
            turnaround = request.completion_time - request.arrival_time

            print(
                f"{request.request_id}: "
                f"arrival={request.arrival_time}, "
                f"start={request.start_time}, "
                f"finish={request.completion_time}, "
                f"generated={request.generated_tokens}, "
                f"waiting={waiting_time}, "
                f"turnaround={turnaround}"
            )

        utilization = (
            100 * self.busy_slots / self.total_slot_capacity
            if self.total_slot_capacity
            else 0
        )

        print(f"\nTotal simulated time: {self.time} steps")
        print(f"Completed requests: {len(self.completed)}")
        print(f"Active-slot utilization: {utilization:.1f}%")


def main():
    scheduler = ContinuousBatchScheduler(max_active_requests=2)

    # Request ID, arrival time, requested output length.
    requests = [
        Request("A", arrival_time=0, max_new_tokens=2),
        Request("B", arrival_time=0, max_new_tokens=5),
        Request("C", arrival_time=1, max_new_tokens=3),
        Request("D", arrival_time=2, max_new_tokens=2),
    ]

    for request in requests:
        scheduler.submit(request)

    scheduler.run()

    # Basic correctness checks.
    assert len(scheduler.completed) == 4
    assert all(r.finished for r in scheduler.completed)

    by_id = {r.request_id: r for r in scheduler.completed}

    assert by_id["A"].completion_time == 2
    assert by_id["B"].completion_time == 5
    assert by_id["C"].start_time == 2
    assert by_id["D"].start_time == 5

    print("\nPASS: All scheduler correctness checks passed.")


if __name__ == "__main__":
    main()
