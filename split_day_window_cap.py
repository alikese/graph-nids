import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", required=True)
    parser.add_argument("--packet-cap", type=int, required=True)
    parser.add_argument("--window-seconds", type=int, default=10)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def clean_output(output_dir, day):
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob(f"{day}-ip_*.csv"):
        path.unlink()


def main():
    args = parse_args()
    if args.packet_cap <= 0:
        raise ValueError("packet-cap must be positive")
    if args.window_seconds <= 0:
        raise ValueError("window-seconds must be positive")

    input_path = args.input or ROOT / "data" / f"{args.day}-ip.csv"
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    clean_output(args.output_dir, args.day)

    handle = None
    writer = None
    temporary_path = None
    bucket_start = None
    packet_count = 0
    total_packets = 0
    window_count = 0
    first_timestamp = 0.0
    last_timestamp = 0.0

    def finish_window(stop_reason):
        nonlocal handle, writer, temporary_path
        if handle is None:
            return
        handle.close()
        final_name = (
            f"{args.day}-ip_{window_count:06d}_"
            f"{round(first_timestamp * 1_000_000)}_"
            f"{round(last_timestamp * 1_000_000)}_"
            f"{stop_reason}.csv"
        )
        temporary_path.replace(args.output_dir / final_name)
        handle = None
        writer = None
        temporary_path = None

    with input_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
        buffering=1024 * 1024,
    ) as source:
        header = source.readline()
        columns = next(csv.reader([header]))
        timestamp_index = columns.index("timestamp")
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

            time_limit_reached = (
                handle is not None and current_bucket != bucket_start
            )
            packet_limit_reached = (
                handle is not None and packet_count >= args.packet_cap
            )
            if handle is None or time_limit_reached or packet_limit_reached:
                if time_limit_reached:
                    finish_window("time")
                elif packet_limit_reached:
                    finish_window("cap")
                bucket_start = current_bucket
                packet_count = 0
                window_count += 1
                first_timestamp = timestamp
                last_timestamp = timestamp
                temporary_path = (
                    args.output_dir
                    / f"{args.day}-ip_{window_count:06d}_pending.csv"
                )
                handle = temporary_path.open(
                    "w",
                    encoding="utf-8",
                    newline="",
                    buffering=1024 * 1024,
                )
                writer = handle.write
                writer(header)

            writer(line)
            packet_count += 1
            total_packets += 1
            last_timestamp = max(last_timestamp, timestamp)

            if total_packets % 1_000_000 == 0:
                print(
                    f"day={args.day} cap={args.packet_cap} "
                    f"packets={total_packets} windows={window_count}",
                    flush=True,
                )

    finish_window("eof")
    print(
        f"done day={args.day} cap={args.packet_cap} "
        f"packets={total_packets} windows={window_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()
