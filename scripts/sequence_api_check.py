#!/usr/bin/env python3

import argparse
import pathlib
import statistics
import sys
import time


def _bench(fn, reps: int) -> tuple[float, float]:
    times_ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    if len(times_ms) == 1:
        return times_ms[0], 0.0
    return statistics.mean(times_ms), statistics.stdev(times_ms)


def _pick_corpus_seq(engine, target_len: int) -> list[int]:
    for doc_ix in [0, 1, 2, 3, 4, 10, 20, 50, 100, 200, 500]:
        doc = engine.get_doc_by_ix(doc_ix=doc_ix, max_disp_len=max(512, target_len * 2))
        if "error" in doc:
            continue
        tokens = [t for t in doc["token_ids"] if t != 65535]
        if len(tokens) >= target_len:
            return tokens[:target_len]
    raise RuntimeError(f"Could not find a doc with at least {target_len} tokens.")


def _assert_prob_seq_equal(engine, seq: list[int]) -> None:
    seq_results = engine.prob_sequence(seq)
    loop_results = [engine.prob(prompt_ids=seq[:i], cont_id=seq[i]) for i in range(len(seq))]
    for i, (a, b) in enumerate(zip(seq_results, loop_results)):
        if a != b:
            raise AssertionError(f"prob mismatch at position {i}: {a} != {b}")


def _assert_infgram_prob_seq_equal(engine, seq: list[int]) -> None:
    seq_results = engine.infgram_prob_sequence(seq)
    loop_results = [engine.infgram_prob(prompt_ids=seq[:i], cont_id=seq[i]) for i in range(len(seq))]
    for i, (a, b) in enumerate(zip(seq_results, loop_results)):
        if a != b:
            raise AssertionError(f"infgram_prob mismatch at position {i}: {a} != {b}")


def _assert_batched_equal(engine, seq: list[int]) -> None:
    batch = [seq[: max(1, len(seq) // 4)], seq[: max(2, len(seq) // 2)], seq]

    prob_seq_batch = engine.prob_batched_sequence(batch)
    prob_loop_batch = [[engine.prob(prompt_ids=s[:i], cont_id=s[i]) for i in range(len(s))] for s in batch]
    if prob_seq_batch != prob_loop_batch:
        raise AssertionError("prob_batched_sequence mismatch")

    inf_seq_batch = engine.infgram_prob_batched_sequence(batch)
    inf_loop_batch = [[engine.infgram_prob(prompt_ids=s[:i], cont_id=s[i]) for i in range(len(s))] for s in batch]
    if inf_seq_batch != inf_loop_batch:
        raise AssertionError("infgram_prob_batched_sequence mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description="Correctness + perf check for sequence APIs.")
    parser.add_argument("--index-dir", required=True, help="Path to index dir (e.g. .../index/v4_pileval_llama)")
    parser.add_argument("--eos-token-id", type=int, default=2)
    parser.add_argument("--token-dtype", default="u16", choices=["u8", "u16", "u32"])
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--reps", type=int, default=7)
    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "pkg"))
    from infini_gram.engine import InfiniGramEngine

    engine = InfiniGramEngine(
        s3_names=[],
        index_dir=args.index_dir,
        eos_token_id=args.eos_token_id,
        token_dtype=args.token_dtype,
    )

    short_seq = [5613, 4086, 9068] * 8
    corpus_seq = _pick_corpus_seq(engine, args.seq_len)

    _assert_prob_seq_equal(engine, short_seq)
    _assert_infgram_prob_seq_equal(engine, short_seq)
    _assert_prob_seq_equal(engine, corpus_seq)
    _assert_infgram_prob_seq_equal(engine, corpus_seq)
    _assert_batched_equal(engine, short_seq)
    _assert_batched_equal(engine, corpus_seq)

    prob_loop = lambda s: [engine.prob(prompt_ids=s[:i], cont_id=s[i]) for i in range(len(s))]
    prob_seq = lambda s: engine.prob_sequence(s)
    inf_loop = lambda s: [engine.infgram_prob(prompt_ids=s[:i], cont_id=s[i]) for i in range(len(s))]
    inf_seq = lambda s: engine.infgram_prob_sequence(s)

    short_prob_loop, short_prob_loop_sd = _bench(lambda: prob_loop(short_seq), args.reps)
    short_prob_seq, short_prob_seq_sd = _bench(lambda: prob_seq(short_seq), args.reps)
    short_inf_loop, short_inf_loop_sd = _bench(lambda: inf_loop(short_seq), args.reps)
    short_inf_seq, short_inf_seq_sd = _bench(lambda: inf_seq(short_seq), args.reps)

    long_prob_loop, long_prob_loop_sd = _bench(lambda: prob_loop(corpus_seq), args.reps)
    long_prob_seq, long_prob_seq_sd = _bench(lambda: prob_seq(corpus_seq), args.reps)
    long_inf_loop, long_inf_loop_sd = _bench(lambda: inf_loop(corpus_seq), args.reps)
    long_inf_seq, long_inf_seq_sd = _bench(lambda: inf_seq(corpus_seq), args.reps)

    print("all correctness checks passed")
    print()
    print("short sequence length:", len(short_seq))
    print(f"prob loop: {short_prob_loop:.3f} ms (sd {short_prob_loop_sd:.3f})")
    print(f"prob sequence: {short_prob_seq:.3f} ms (sd {short_prob_seq_sd:.3f}) speedup={short_prob_loop / short_prob_seq:.3f}x")
    print(f"infgram loop: {short_inf_loop:.3f} ms (sd {short_inf_loop_sd:.3f})")
    print(f"infgram sequence: {short_inf_seq:.3f} ms (sd {short_inf_seq_sd:.3f}) speedup={short_inf_loop / short_inf_seq:.3f}x")
    print()
    print("corpus sequence length:", len(corpus_seq))
    print(f"prob loop: {long_prob_loop:.3f} ms (sd {long_prob_loop_sd:.3f})")
    print(f"prob sequence: {long_prob_seq:.3f} ms (sd {long_prob_seq_sd:.3f}) speedup={long_prob_loop / long_prob_seq:.3f}x")
    print(f"infgram loop: {long_inf_loop:.3f} ms (sd {long_inf_loop_sd:.3f})")
    print(f"infgram sequence: {long_inf_seq:.3f} ms (sd {long_inf_seq_sd:.3f}) speedup={long_inf_loop / long_inf_seq:.3f}x")


if __name__ == "__main__":
    main()
