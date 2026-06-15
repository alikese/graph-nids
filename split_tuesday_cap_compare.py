import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "Tuesday-ip.csv"


@dataclass
class SplitState:
    cap: int
    output_dir: Path
    header: str
    writer: object = None
    handle: object = None
    bucket_start: int = None
    packet_count: int = 0
    window_count: int = 0
    total_packets: int = 0
    first_timestamp: float = 0.0
    last_timestamp: float = 0.0

    def close_window(self):
        if self.handle is None:
            return
        self.handle.close()
        self.handle = None
        self.writer = None

    def open_window(self, bucket_start, timestamp):
        self.close_window()
        self.bucket_start = bucket_start
        self.packet_count = 0
        self.window_count += 1
        self.first_timestamp = timestamp
        self.last_timestamp = timestamp
        filename = (
            f"Tuesday-ip_{self.window_count:06d}_"
            f"{round(timestamp * 1_000_000)}_pending.csv"
        )
        self.handle = (self.output_dir / filename).open(
            "w",
            encoding="utf-8",
            newline="",
            buffering=1024 * 1024,
        )
        self.writer = self.handle.write
        self.writer(self.header)

    def finish_window(self):
        if self.handle is None:
            return
        path = Path(self.handle.name)
        final_name = (
            f"Tuesday-ip_{self.window_count:06d}_"
            f"{round(self.first_timestamp * 1_000_000)}_"
            f"{round(self.last_timestamp * 1_000_000)}.csv"
        )
        self.close_window()
        path.replace(self.output_dir / final_name)

    def add(self, line, timestamp, window_seconds):
        bucket_start = int(timestamp // window_seconds) * window_seconds
        if (
            self.handle is None
            or bucket_start != self.bucket_start
            or self.packet_count >= self.cap
        ):
            self.finish_window()
            self.open_window(bucket_start, timestamp)
        self.writer(line)
        self.packet_count += 1
        self.total_packets += 1
        self.last_timestamp = max(self.last_timestamp, timestamp)


def prepare_output(path):
    path.mkdir(parents=True, exist_ok=True)
    for file_path in path.glob("Tuesday-ip_*.csv"):
        file_path.unlink()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--window-seconds", type=int, default=10)
    parser.add_argument(
        "--output-5000",
        type=Path,
        default=ROOT / "windows Tuesday cap5000",
    )
    parser.add_argument(
        "--output-10000",
        type=Path,
        default=ROOT / "windows Tuesday cap10000",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.window_seconds <= 0:
        raise ValueError("window-seconds must be positive")
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    prepare_output(args.output_5000)
    prepare_output(args.output_10000)

    with args.input.open(
        "r",
        encoding="utf-8-sig",
        newline="",
        buffering=1024 * 1024,
    ) as source:
        header = source.readline()
        columns = next(csv.reader([header]))
        timestamp_index = columns.index("timestamp")
        states = (
            SplitState(5000, args.output_5000, header),
            SplitState(10000, args.output_10000, header),
        )
        previous_bucket = None
        for line_number, line in enumerate(source, start=2):
            fields = line.split(",")
            timestamp = float(fields[timestamp_index])
            current_bucket = (
                int(timestamp // args.window_seconds)
                * args.window_seconds
            )
            if (
                previous_bucket is not None
                and current_bucket < previous_bucket
            ):
                raise ValueError(
                    f"10-second buckets are not sorted at line {line_number}"
                )
            previous_bucket = current_bucket
            for state in states:
                state.add(line, timestamp, args.window_seconds)
            if line_number % 1_000_000 == 0:
                print(
                    f"rows={line_number - 1} "
                    f"cap5000_windows={states[0].window_count} "
                    f"cap10000_windows={states[1].window_count}",
                    flush=True,
                )

    for state in states:
        state.finish_window()
        print(
            f"cap={state.cap} windows={state.window_count} "
            f"packets={state.total_packets} output={state.output_dir}",
            flush=True,
        )


if __name__ == "__main__":
    main()
