#!/usr/bin/env python3

import argparse
import pathlib
import random
import statistics
import sys
import time


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bench(fn, reps: int) -> tuple[float, float]:
    times_ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    if len(times_ms) == 1:
        return times_ms[0], 0.0
    return statistics.mean(times_ms), statistics.stdev(times_ms)


def _pick_corpus_seq(engine, target_len: int, eos_token_id: int) -> list[int]:
    """Extract a token sequence of *target_len* from the index."""
    for doc_ix in [0, 1, 2, 3, 4, 10, 20, 50, 100, 200, 500]:
        doc = engine.get_doc_by_ix(doc_ix=doc_ix, max_disp_len=max(512, target_len * 2))
        if "error" in doc:
            continue
        tokens = [t for t in doc["token_ids"] if t != eos_token_id]
        if len(tokens) >= target_len:
            return tokens[:target_len]
    raise RuntimeError(f"Could not find a doc with at least {target_len} tokens.")


def _mine_corpus_segments(engine, eos_token_id: int) -> list[tuple[str, list[int]]]:
    """Extract 20-30 segments from different documents at varied lengths."""
    doc_indices = [0, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    target_lengths = [16, 64, 256, 1024]
    segments = []
    for doc_ix in doc_indices:
        doc = engine.get_doc_by_ix(doc_ix=doc_ix, max_disp_len=4096)
        if "error" in doc:
            continue
        tokens = [t for t in doc["token_ids"] if t != eos_token_id]
        if len(tokens) < 8:
            continue
        for tgt in target_lengths:
            if len(tokens) >= tgt:
                segments.append((f"doc{doc_ix}_len{tgt}", tokens[:tgt]))
    return segments


def _build_zero_count_seq(engine, eos_token_id: int) -> tuple[list[int], int]:
    """Construct a sequence that hits count==0 at a known position.

    Strategy: take a short corpus prefix, then extend with random tokens
    until count drops to zero.  After the zero point, append common tokens
    so infgram can potentially recover via failure links.
    """
    rng = random.Random(123)
    doc = engine.get_doc_by_ix(doc_ix=0, max_disp_len=512)
    assert "error" not in doc, f"Cannot read doc 0: {doc}"
    tokens = [t for t in doc["token_ids"] if t != eos_token_id]
    prefix = tokens[:8]

    tid_max = engine.token_id_max

    # Extend token by token until count == 0
    for step in range(200):
        cand = rng.randint(0, min(tid_max, 50000))
        trial = prefix + [cand]
        cnt = engine.count(trial)
        if cnt.get("count", cnt.get("cnt", 1)) == 0:
            first_zero_pos = len(trial) - 1  # position of last token in *trial* as a sequence
            # Append 8 common tokens so infgram can recover
            common_tail = [t for t in range(1, 9) if t <= tid_max]
            zero_seq = trial + common_tail
            # first_zero_pos is the index into zero_seq where prob_sequence will first report 0
            # In prob_sequence, position i corresponds to P(seq[i] | seq[:i]).
            # The zero count means count(seq[:first_zero_pos+1]) == 0, but what matters
            # is the prompt count at each position.
            # Position first_zero_pos: prompt = trial[:-1], cont = trial[-1].
            # prompt_cnt = count(trial[:-1]) which should be > 0, but cont_cnt = count(trial) == 0.
            # The cascade starts at first_zero_pos + 1 because that's when prompt_cnt becomes 0.
            # Actually let's verify: count(trial) == 0 means prompt_cnt at position first_zero_pos+1
            # (where prompt = trial, cont = common_tail[0]) will be 0.
            return zero_seq, first_zero_pos + 1
        prefix = trial

    # If we never hit zero (unlikely), just return a synthetic sequence
    # that almost certainly won't be in any corpus
    synth = [t % (tid_max + 1) for t in
             [12345, 54321, 11111, 22222, 33333, 44444, 55555, 60000,
              1, 2, 3, 4, 5, 6, 7, 8]]
    # Find where it actually hits zero
    for i in range(1, len(synth)):
        cnt = engine.count(synth[:i + 1])
        if cnt.get("count", cnt.get("cnt", 1)) == 0:
            return synth, i + 1
    return synth, 4  # fallback guess


# ---------------------------------------------------------------------------
# Correctness checks
# ---------------------------------------------------------------------------

def _assert_prob_seq_equal(engine, seq: list[int], label: str = "") -> None:
    """prob_sequence must match the loop-based reference."""
    seq_results = engine.prob_sequence(seq)
    for i in range(len(seq)):
        expected = engine.prob(prompt_ids=seq[:i], cont_id=seq[i])
        actual = seq_results[i]
        assert actual == expected, (
            f"prob mismatch [{label}] at position {i}/{len(seq)}: "
            f"expected {expected}, got {actual}"
        )


def _assert_infgram_prob_seq_equal(engine, seq: list[int], label: str = "") -> None:
    """infgram_prob_sequence must match the loop-based reference."""
    seq_results = engine.infgram_prob_sequence(seq)
    for i in range(len(seq)):
        expected = engine.infgram_prob(prompt_ids=seq[:i], cont_id=seq[i])
        actual = seq_results[i]
        assert actual == expected, (
            f"infgram_prob mismatch [{label}] at position {i}/{len(seq)}: "
            f"expected {expected}, got {actual}"
        )


def _assert_cross_validate_position_zero(engine, seq: list[int], label: str = "") -> None:
    """At position 0, prob and infgram_prob must agree (empty context)."""
    if len(seq) == 0:
        return
    prob_results = engine.prob_sequence(seq)
    infgram_results = engine.infgram_prob_sequence(seq)
    p = prob_results[0]
    ig = infgram_results[0]
    assert p['prompt_cnt'] == ig['prompt_cnt'], (
        f"pos0 prompt_cnt mismatch [{label}]: prob={p['prompt_cnt']}, infgram={ig['prompt_cnt']}"
    )
    assert p['cont_cnt'] == ig['cont_cnt'], (
        f"pos0 cont_cnt mismatch [{label}]: prob={p['cont_cnt']}, infgram={ig['cont_cnt']}"
    )
    assert p['prob'] == ig['prob'], (
        f"pos0 prob mismatch [{label}]: prob={p['prob']}, infgram={ig['prob']}"
    )
    assert ig['suffix_len'] == 0, (
        f"pos0 suffix_len should be 0 [{label}]: got {ig['suffix_len']}"
    )


def _assert_zero_count_behavior(engine, zero_seq: list[int], first_zero_pos: int) -> None:
    """Verify the zero-count cascade in prob_sequence and recovery in infgram."""
    prob_results = engine.prob_sequence(zero_seq)
    for i in range(first_zero_pos, len(prob_results)):
        r = prob_results[i]
        assert r == {'prompt_cnt': 0, 'cont_cnt': 0, 'prob': -1.0}, (
            f"zero-count cascade failed at position {i}: expected all-zero, got {r}"
        )

    # infgram_prob_sequence can recover via failure links
    infgram_results = engine.infgram_prob_sequence(zero_seq)
    recovered = any(r['prob'] != -1.0 for r in infgram_results[first_zero_pos:])
    if recovered:
        first_recovery = next(
            i for i in range(first_zero_pos, len(infgram_results))
            if infgram_results[i]['prob'] != -1.0
        )
        print(f"  infgram recovered at position {first_recovery} "
              f"(first_zero={first_zero_pos}, len={len(zero_seq)})")
    else:
        print(f"  infgram did not recover after position {first_zero_pos} "
              f"(corpus may lack matching suffixes)")


def _assert_suffix_len_valid(engine, seq: list[int], label: str = "",
                             skip_maximality: bool = False) -> None:
    """Validate suffix_len semantics for infgram_prob_sequence results."""
    results = engine.infgram_prob_sequence(seq)
    for i, r in enumerate(results):
        sl = r['suffix_len']
        assert sl >= 0, (
            f"suffix_len negative [{label}] at pos {i}: {sl}"
        )
        assert sl <= i, (
            f"suffix_len > i [{label}] at pos {i}: suffix_len={sl}, i={i}"
        )
        if i == 0:
            assert sl == 0, (
                f"suffix_len at pos 0 should be 0 [{label}]: got {sl}"
            )

        if skip_maximality:
            continue

        # If suffix_len > 0 and prompt_cnt > 0, the suffix must exist in the corpus
        if sl > 0 and r['prompt_cnt'] > 0:
            suffix = seq[i - sl:i]
            cnt = engine.count(suffix)
            cnt_val = cnt.get("count", cnt.get("cnt", 0))
            assert cnt_val > 0, (
                f"suffix_len validity [{label}] at pos {i}: "
                f"suffix seq[{i - sl}:{i}] has count 0 but suffix_len={sl}"
            )

        # Maximality: one more token of context should have count 0
        if sl < i and i > 0 and r['prompt_cnt'] > 0:
            longer_suffix = seq[i - sl - 1:i]
            cnt = engine.count(longer_suffix)
            cnt_val = cnt.get("count", cnt.get("cnt", 0))
            assert cnt_val == 0, (
                f"suffix_len maximality [{label}] at pos {i}: "
                f"seq[{i - sl - 1}:{i}] has count {cnt_val} > 0 but suffix_len={sl} < i={i}"
            )


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

def _test_edge_cases(engine) -> None:
    print("  edge cases: empty sequence")
    assert engine.prob_sequence([]) == [], "prob_sequence([]) should return []"
    assert engine.infgram_prob_sequence([]) == [], "infgram_prob_sequence([]) should return []"

    print("  edge cases: single token")
    tok = min(1, engine.token_id_max)
    single_prob = engine.prob_sequence([tok])
    single_ref = engine.prob(prompt_ids=[], cont_id=tok)
    assert len(single_prob) == 1, f"prob_sequence([{tok}]) should return 1 result"
    assert single_prob[0] == single_ref, (
        f"single-token prob mismatch: {single_prob[0]} != {single_ref}"
    )
    single_inf = engine.infgram_prob_sequence([tok])
    single_inf_ref = engine.infgram_prob(prompt_ids=[], cont_id=tok)
    assert len(single_inf) == 1, f"infgram_prob_sequence([{tok}]) should return 1 result"
    assert single_inf[0] == single_inf_ref, (
        f"single-token infgram mismatch: {single_inf[0]} != {single_inf_ref}"
    )

    print("  edge cases: two tokens")
    tok2 = min(2, engine.token_id_max)
    seq2 = [tok, tok2]
    _assert_prob_seq_equal(engine, seq2, label="two-token")
    _assert_infgram_prob_seq_equal(engine, seq2, label="two-token")

    print("  edge cases: batched([]) and batched([[]])")
    assert engine.prob_batched_sequence([]) == [], "prob_batched_sequence([]) should return []"
    assert engine.infgram_prob_batched_sequence([]) == [], (
        "infgram_prob_batched_sequence([]) should return []"
    )
    assert engine.prob_batched_sequence([[]]) == [[]], (
        "prob_batched_sequence([[]]) should return [[]]"
    )
    assert engine.infgram_prob_batched_sequence([[]]) == [[]], (
        "infgram_prob_batched_sequence([[]]) should return [[]]"
    )


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

def _test_error_handling(engine) -> None:
    print("  error handling: invalid inputs")

    for name, fn in [
        ("prob_sequence", engine.prob_sequence),
        ("infgram_prob_sequence", engine.infgram_prob_sequence),
    ]:
        result = fn("not a list")
        assert isinstance(result, dict) and 'error' in result, (
            f"{name}('not a list') should return error dict, got {result}"
        )
        result = fn([-1])
        assert isinstance(result, dict) and 'error' in result, (
            f"{name}([-1]) should return error dict, got {result}"
        )

    for name, fn in [
        ("prob_batched_sequence", engine.prob_batched_sequence),
        ("infgram_prob_batched_sequence", engine.infgram_prob_batched_sequence),
    ]:
        result = fn([[-1]])
        assert isinstance(result, dict) and 'error' in result, (
            f"{name}([[-1]]) should return error dict, got {result}"
        )


# ---------------------------------------------------------------------------
# Batch tests
# ---------------------------------------------------------------------------

def _assert_batched_equal(engine, batch: list[list[int]], label: str = "") -> None:
    """batched(batch) must equal [sequence(s) for s in batch]."""
    prob_batched = engine.prob_batched_sequence(batch)
    prob_individual = [engine.prob_sequence(s) for s in batch]
    for batch_idx, (actual, expected) in enumerate(zip(prob_batched, prob_individual)):
        for pos, (a, e) in enumerate(zip(actual, expected)):
            assert a == e, (
                f"prob_batched mismatch [{label}] batch_item={batch_idx} pos={pos}: "
                f"expected {e}, got {a}"
            )

    inf_batched = engine.infgram_prob_batched_sequence(batch)
    inf_individual = [engine.infgram_prob_sequence(s) for s in batch]
    for batch_idx, (actual, expected) in enumerate(zip(inf_batched, inf_individual)):
        for pos, (a, e) in enumerate(zip(actual, expected)):
            assert a == e, (
                f"infgram_batched mismatch [{label}] batch_item={batch_idx} pos={pos}: "
                f"expected {e}, got {a}"
            )


# ---------------------------------------------------------------------------
# Fuzz testing
# ---------------------------------------------------------------------------

def _seq_summary(seq: list[int], max_show: int = 20) -> str:
    if len(seq) <= max_show * 2:
        return str(seq)
    return f"{seq[:max_show]}...{seq[-max_show:]} (len={len(seq)})"


def _run_fuzz_tier1(engine, rng: random.Random, token_id_max: int,
                    eos_token_id: int) -> None:
    """Tier 1: breadth — geometric lengths, 5 random + 1 corpus-mined each."""
    lengths = [32, 128, 512, 2048, 8192]
    for length in lengths:
        for trial in range(5):
            seq = [rng.randint(0, token_id_max) for _ in range(length)]
            label = f"fuzz_t1_rand_len{length}_trial{trial}"
            try:
                _assert_prob_seq_equal(engine, seq, label=label)
                _assert_infgram_prob_seq_equal(engine, seq, label=label)
            except AssertionError:
                print(f"  FAIL {label}: seed=42, seq={_seq_summary(seq)}")
                raise
        # Corpus-mined at this length
        try:
            corpus_seg = _pick_corpus_seq(engine, length, eos_token_id)
            label = f"fuzz_t1_corpus_len{length}"
            _assert_prob_seq_equal(engine, corpus_seg, label=label)
            _assert_infgram_prob_seq_equal(engine, corpus_seg, label=label)
        except RuntimeError:
            pass  # corpus doesn't have a doc this long
        print(f"    tier1 length={length}: OK")


def _run_fuzz_tier2(engine, rng: random.Random, token_id_max: int) -> list[tuple[str, list[int]]]:
    """Tier 2: stress — large sequences. Returns sequences for benchmarking."""
    lengths = [16384, 32768, 65536]
    stress_seqs = []
    for length in lengths:
        for trial in range(2):
            seq = [rng.randint(0, token_id_max) for _ in range(length)]
            label = f"fuzz_t2_len{length}_trial{trial}"
            try:
                _assert_prob_seq_equal(engine, seq, label=label)
                _assert_infgram_prob_seq_equal(engine, seq, label=label)
            except AssertionError:
                print(f"  FAIL {label}: seed=42, seq={_seq_summary(seq)}")
                raise
            stress_seqs.append((label, seq))
        print(f"    tier2 length={length}: OK")
    return stress_seqs


def _run_fuzz_tier3(engine, rng: random.Random, token_id_max: int,
                    timeout_s: float, eos_token_id: int) -> int:
    """Tier 3: adaptive ceiling — double length until timeout.
    Returns the maximum length reached."""
    length = 1024
    max_reached = 0
    while True:
        seq = [rng.randint(0, token_id_max) for _ in range(length)]

        t0 = time.perf_counter()
        engine.prob_sequence(seq)
        elapsed_prob = time.perf_counter() - t0

        t0 = time.perf_counter()
        engine.infgram_prob_sequence(seq)
        elapsed_inf = time.perf_counter() - t0

        max_elapsed = max(elapsed_prob, elapsed_inf)
        print(f"    tier3 length={length}: prob={elapsed_prob:.3f}s infgram={elapsed_inf:.3f}s")

        if max_elapsed > timeout_s:
            print(f"    tier3 ceiling reached at length={length} ({max_elapsed:.1f}s > {timeout_s}s)")
            max_reached = length
            break

        max_reached = length

        # Also try a corpus-mined sequence at this length
        try:
            corpus_seg = _pick_corpus_seq(engine, length, eos_token_id)
            t0 = time.perf_counter()
            engine.prob_sequence(corpus_seg)
            t1 = time.perf_counter()
            engine.infgram_prob_sequence(corpus_seg)
            t2 = time.perf_counter()
            print(f"    tier3 corpus length={length}: prob={t1-t0:.3f}s infgram={t2-t1:.3f}s")
        except RuntimeError:
            pass

        length *= 2
        if length > 1_000_000:
            print(f"    tier3 stopped at 1M tokens (no timeout hit)")
            break

    return max_reached


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Correctness + perf check for sequence APIs.")
    parser.add_argument("--index-dir", required=True, help="Path to index dir")
    parser.add_argument("--eos-token-id", type=int, default=2)
    parser.add_argument("--token-dtype", default="u16", choices=["u8", "u16", "u32"])
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--skip-fuzz", action="store_true",
                        help="Skip Tier 2/3 fuzz tests for faster runs")
    parser.add_argument("--fuzz-timeout", type=float, default=10.0,
                        help="Time threshold in seconds for Tier 3 adaptive ceiling")
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

    # -----------------------------------------------------------------------
    # Test sequences (clamped to valid token range)
    # -----------------------------------------------------------------------
    eos_token_id = args.eos_token_id
    tid_max = engine.token_id_max
    diverse_seq = [t % (tid_max + 1) for t in [
        10, 257, 1023, 3000, 5000, 7777, 9999, 12000,
        15000, 18000, 21000, 24000, 27000, 30000, 33000, 36000,
        39000, 42000, 45000, 48000, 50000, 52000, 54000, 56000,
    ]]
    repeat_seq = [t % (tid_max + 1) for t in [5613, 4086, 9068]] * 8
    corpus_seq = _pick_corpus_seq(engine, args.seq_len, eos_token_id)

    # -----------------------------------------------------------------------
    # A. Fixed sequence correctness
    # -----------------------------------------------------------------------
    print("=== Fixed sequence correctness ===")
    for label, seq in [("diverse", diverse_seq), ("repeat", repeat_seq), ("corpus", corpus_seq)]:
        print(f"  {label} (len={len(seq)})")
        _assert_prob_seq_equal(engine, seq, label=label)
        _assert_infgram_prob_seq_equal(engine, seq, label=label)
        _assert_cross_validate_position_zero(engine, seq, label=label)
    print("  PASSED")

    # -----------------------------------------------------------------------
    # B+C. Zero-count cascade + recovery
    # -----------------------------------------------------------------------
    print("=== Zero-count cascade ===")
    zero_seq, first_zero_pos = _build_zero_count_seq(engine, eos_token_id)
    print(f"  zero_seq len={len(zero_seq)}, first_zero_pos={first_zero_pos}")
    _assert_prob_seq_equal(engine, zero_seq, label="zero_count")
    _assert_infgram_prob_seq_equal(engine, zero_seq, label="zero_count")
    _assert_zero_count_behavior(engine, zero_seq, first_zero_pos)
    print("  PASSED")

    # -----------------------------------------------------------------------
    # D. Cross-validation at position 0 (already done above, also on zero_seq)
    # -----------------------------------------------------------------------
    _assert_cross_validate_position_zero(engine, zero_seq, label="zero_count")

    # -----------------------------------------------------------------------
    # E. Edge cases
    # -----------------------------------------------------------------------
    print("=== Edge cases ===")
    _test_edge_cases(engine)
    print("  PASSED")

    # -----------------------------------------------------------------------
    # Error handling
    # -----------------------------------------------------------------------
    print("=== Error handling ===")
    _test_error_handling(engine)
    print("  PASSED")

    # -----------------------------------------------------------------------
    # F. Batch tests
    # -----------------------------------------------------------------------
    print("=== Batch tests ===")
    # Independent sequences
    independent_batch = [diverse_seq, repeat_seq[:12], corpus_seq[:20]]
    _assert_batched_equal(engine, independent_batch, label="independent")
    print("  independent batch: PASSED")

    # Overlapping-prefix pattern (legacy)
    overlap_batch = [
        corpus_seq[:max(1, len(corpus_seq) // 4)],
        corpus_seq[:max(2, len(corpus_seq) // 2)],
        corpus_seq,
    ]
    _assert_batched_equal(engine, overlap_batch, label="overlapping")
    print("  overlapping batch: PASSED")

    # -----------------------------------------------------------------------
    # G. suffix_len validation
    # -----------------------------------------------------------------------
    print("=== suffix_len validation ===")
    for label, seq in [("diverse", diverse_seq), ("repeat", repeat_seq),
                       ("corpus", corpus_seq), ("zero_count", zero_seq)]:
        _assert_suffix_len_valid(engine, seq, label=label)
        print(f"  {label}: OK")
    print("  PASSED")

    # -----------------------------------------------------------------------
    # Corpus-mined battery
    # -----------------------------------------------------------------------
    print("=== Corpus-mined battery ===")
    corpus_segments = _mine_corpus_segments(engine, eos_token_id)
    print(f"  mined {len(corpus_segments)} segments")
    for label, seg in corpus_segments:
        _assert_prob_seq_equal(engine, seg, label=label)
        _assert_infgram_prob_seq_equal(engine, seg, label=label)
        _assert_cross_validate_position_zero(engine, seg, label=label)
    print("  PASSED")

    # -----------------------------------------------------------------------
    # H. Fuzz testing
    # -----------------------------------------------------------------------
    print("=== Fuzz testing ===")
    rng = random.Random(42)
    token_id_max = min(engine.token_id_max, 50000)  # cap for realistic token IDs

    print("  tier 1: breadth")
    _run_fuzz_tier1(engine, rng, token_id_max, eos_token_id)
    print("  tier 1: PASSED")

    stress_seqs = []
    if not args.skip_fuzz:
        print("  tier 2: stress")
        stress_seqs = _run_fuzz_tier2(engine, rng, token_id_max)
        print("  tier 2: PASSED")

        print("  tier 3: adaptive ceiling")
        max_len = _run_fuzz_tier3(engine, rng, token_id_max, args.fuzz_timeout, eos_token_id)
        print(f"  tier 3: max length reached = {max_len}")
    else:
        print("  tier 2/3: SKIPPED (--skip-fuzz)")

    # -----------------------------------------------------------------------
    # K. Benchmarks
    # -----------------------------------------------------------------------
    print()
    print("=== Benchmarks ===")

    prob_loop = lambda s: [engine.prob(prompt_ids=s[:i], cont_id=s[i]) for i in range(len(s))]
    prob_seq = lambda s: engine.prob_sequence(s)
    inf_loop = lambda s: [engine.infgram_prob(prompt_ids=s[:i], cont_id=s[i]) for i in range(len(s))]
    inf_seq = lambda s: engine.infgram_prob_sequence(s)

    for label, seq in [("diverse", diverse_seq), ("corpus", corpus_seq)]:
        print(f"\n  {label} (len={len(seq)}):")
        pl_mean, pl_sd = _bench(lambda s=seq: prob_loop(s), args.reps)
        ps_mean, ps_sd = _bench(lambda s=seq: prob_seq(s), args.reps)
        il_mean, il_sd = _bench(lambda s=seq: inf_loop(s), args.reps)
        is_mean, is_sd = _bench(lambda s=seq: inf_seq(s), args.reps)
        print(f"    prob loop:     {pl_mean:10.3f} ms (sd {pl_sd:.3f})")
        print(f"    prob sequence: {ps_mean:10.3f} ms (sd {ps_sd:.3f})  speedup={pl_mean / ps_mean:.1f}x")
        print(f"    infgram loop:     {il_mean:10.3f} ms (sd {il_sd:.3f})")
        print(f"    infgram sequence: {is_mean:10.3f} ms (sd {is_sd:.3f})  speedup={il_mean / is_mean:.1f}x")

    # Benchmark the largest stress sequences from Tier 2
    if stress_seqs:
        # Just benchmark the last (longest) one
        label, seq = stress_seqs[-1]
        print(f"\n  stress {label} (len={len(seq)}):")
        ps_mean, ps_sd = _bench(lambda s=seq: prob_seq(s), max(1, args.reps // 3))
        is_mean, is_sd = _bench(lambda s=seq: inf_seq(s), max(1, args.reps // 3))
        print(f"    prob sequence: {ps_mean:10.3f} ms (sd {ps_sd:.3f})")
        print(f"    infgram sequence: {is_mean:10.3f} ms (sd {is_sd:.3f})")
        # Also benchmark the loop for comparison (just 1 rep — it's slow)
        print(f"    (running loop reference once for speedup comparison...)")
        pl_mean, _ = _bench(lambda s=seq: prob_loop(s), 1)
        il_mean, _ = _bench(lambda s=seq: inf_loop(s), 1)
        print(f"    prob loop:     {pl_mean:10.3f} ms  speedup={pl_mean / ps_mean:.1f}x")
        print(f"    infgram loop:     {il_mean:10.3f} ms  speedup={il_mean / is_mean:.1f}x")

    print()
    print("all checks passed")


if __name__ == "__main__":
    main()
