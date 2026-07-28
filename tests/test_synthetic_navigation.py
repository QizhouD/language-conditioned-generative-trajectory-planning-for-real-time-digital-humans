import json
import random

from gtp.data.synthetic_navigation import generate_example, write_dataset


def test_generator_is_deterministic_and_valid(tmp_path):
    first = generate_example(1, random.Random(9))
    second = generate_example(1, random.Random(9))
    assert first == second
    assert len(first.history) == 20
    assert len(first.future_trajectory) == 30
    assert all(len(point) == 8 for point in first.future_trajectory)

    output = tmp_path / "episodes.jsonl"
    write_dataset(output, episodes=4, seed=3)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    assert all(row["goal"]["behavior"] in {"walk", "wave"} for row in rows)
    assert all(max(abs(point[6]) for point in row["future_trajectory"]) <= 3.01 for row in rows)
