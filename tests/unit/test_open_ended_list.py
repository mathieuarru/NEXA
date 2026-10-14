"""List-native open-ended scoring: no live database or LLM required.

In BioHopR an entity name may contain a comma, and a trailing space encodes
PrimeKG's "(disease)" marker ("myocardial infarction " and "myocardial
infarction" are different diseases).
"""
import json
from pathlib import Path

import pytest

from benchmark.bench_utils import (
    compute_set_metrics,
    csv_to_set,
    entity_key,
    entity_list_to_set,
    rescore_open_ended,
)
from benchmark.benchmark import (
    BENCH_TYPES,
    OpenEndedListAnswer,
    evaluate_correctness,
    extract_model_answer,
)

COMMA_NAME = "dermatitis herpetiformis, familial"


def test_comma_inside_an_entity_name_survives():
    assert entity_list_to_set([COMMA_NAME, "autism"]) == {COMMA_NAME, "autism"}


def test_csv_parser_would_shred_a_list_gold():
    """The CSV parser splits comma-bearing names, so it cannot score list golds."""
    shredded = csv_to_set([COMMA_NAME, "autism"])
    assert COMMA_NAME not in shredded
    assert len(shredded) > 2  # bracket/quote debris from str(list)


def test_a_bare_string_is_one_entity_not_a_csv_line():
    assert entity_list_to_set(COMMA_NAME) == {COMMA_NAME}


def test_trailing_space_is_a_distinct_entity():
    both = entity_list_to_set(["myocardial infarction ", "myocardial infarction"])
    assert len(both) == 2, "the (disease) marker encoding was collapsed"


def test_empty_and_none_answers():
    assert entity_list_to_set(None) == set()
    assert entity_list_to_set([]) == set()
    assert entity_list_to_set(["", "APOE"]) == {"APOE"}


def test_correctness_uses_the_list_parser_on_both_sides():
    gold = [COMMA_NAME, "autism"]
    assert evaluate_correctness("open_ended_list", gold, [COMMA_NAME, "autism"])
    assert evaluate_correctness("open_ended_list", gold, ["autism", COMMA_NAME])
    # The shredded form is not accepted as correct.
    assert not evaluate_correctness(
        "open_ended_list", gold, ["dermatitis herpetiformis", "familial", "autism"]
    )
    # CSV scoring shreds the gold too, so it rejects even the correct answer.
    assert not evaluate_correctness("open_ended", gold, [COMMA_NAME, "autism"])


@pytest.mark.parametrize("payload,expected", [
    ([COMMA_NAME], [COMMA_NAME]),
    ("a single string", ["a single string"]),
    ([1, 2], ["1", "2"]),
])
def test_extraction_shapes(payload, expected):
    answer, text, error = extract_model_answer("open_ended_list", payload, "q")
    assert (answer, text, error) == (expected, None, None)


def test_extraction_rejects_a_non_list_non_string():
    _, _, error = extract_model_answer("open_ended_list", {"a": 1}, "q")
    assert error and "OpenEndedList" in error


def test_schema_requires_a_list():
    ok = OpenEndedListAnswer(thinking="t", answer=[COMMA_NAME])
    assert ok.answer == [COMMA_NAME]
    assert BENCH_TYPES["open_ended_list"]["schema"] is OpenEndedListAnswer
    # Its own prefix keeps --run-all from mixing it with the CSV-gold datasets.
    assert BENCH_TYPES["open_ended_list"]["prefix"] != BENCH_TYPES["open_ended"]["prefix"]


def test_gold_coverage_separates_incomplete_from_over_generating():
    gold = {"a", "b", "c"}
    incomplete = compute_set_metrics(gold, {"a"})
    superset = compute_set_metrics(gold, {"a", "b", "c", "d"})
    # Both are "not an exact match", but only one found every gold answer.
    assert incomplete["recall"] < 1.0
    assert superset["recall"] == 1.0
    assert superset["precision"] < 1.0


def test_names_match_case_insensitively():
    metrics = compute_set_metrics({"nusinersen", "Risdiplam"}, {"Nusinersen", "risdiplam"})
    assert metrics["exact_match"] == 1.0 and metrics["jaccard"] == 1.0
    assert evaluate_correctness("open_ended_list", ["nusinersen"], ["Nusinersen"])
    assert evaluate_correctness("open_ended", "APOE, Systemic sclerosis", ["apoe", "systemic Sclerosis"])


def test_matching_keeps_the_disease_marker_and_other_punctuation():
    # The dataset builder turns BioHopR's trailing-space encoding into "(disease)".
    assert compute_set_metrics({"myocardial infarction (disease)"}, {"Myocardial infarction"})["jaccard"] == 0.0
    # PrimeKG holds both names: only a trailing period is dropped, not a comma.
    assert compute_set_metrics({"Carney complex", "Carney complex,"}, {"carney complex"})["jaccard"] == 0.5


def test_typography_is_not_a_different_answer():
    gold = {"Alzheimer's Disease", "K+ channel", "cell-cell junction", "Ketobemidone  Action Pathway", "gliosarcoma"}
    pred = {"Alzheimer\u2019s disease.", "K\u207a channel", "cell\u2013cell junction", "Ketobemidone Action Pathway",
            "glio\u00adsarcoma"}
    assert compute_set_metrics(gold, pred)["exact_match"] == 1.0
    # A different spelling of the name is still a different answer.
    assert compute_set_metrics({"SEPTIN11"}, {"septin 11"})["jaccard"] == 0.0
    assert compute_set_metrics({"Hodgkin's lymphoma"}, {"Hodgkins lymphoma"})["jaccard"] == 0.0


def test_rescore_updates_stored_scores_and_leaves_failures_alone():
    rec = {"type": "open_ended_list", "expected": ["nusinersen", "risdiplam"],
           "model_answer": ["Nusinersen"], "correct": False,
           "metrics": {**compute_set_metrics({"nusinersen", "risdiplam"}, set()), "gold_covered": 0.0}}
    rescore_open_ended(rec)
    assert rec["metrics"]["jaccard"] == 0.5 and rec["metrics"]["gold_covered"] == 0.0
    rec["model_answer"] = ["Risdiplam", "NUSINERSEN"]
    rescore_open_ended(rec)
    assert rec["correct"] is True and rec["metrics"]["gold_covered"] == 1.0
    # A baseline failure carries no metrics and stays scored 0 downstream.
    failed = {"type": "open_ended", "expected": "APOE", "model_answer": "NO_RESPONSE", "correct": False}
    assert "metrics" not in rescore_open_ended(failed)


DATASET = Path(__file__).resolve().parents[2] / "benchmark" / "datasets" / "biohopr" / "BioHopR_2hop.json"


@pytest.mark.skipif(not DATASET.exists(), reason="run benchmark/datasets/biohopr/build_dataset.py first")
def test_built_dataset_is_runner_shaped_and_lossless():
    items = json.loads(DATASET.read_text(encoding="utf-8"))
    assert len(items) == 7633
    assert all(isinstance(i["answer"], list) and i["answer"] for i in items)
    assert all(isinstance(i["question"], str) and i["question"] for i in items)
    # Comma-bearing names are preserved, not split, by the build step.
    assert sum(1 for i in items for a in i["answer"] if "," in a) == 2991
    # Every answer list is already deduplicated under the scoring parser.
    assert all(len(entity_list_to_set(i["answer"])) == len(i["answer"]) for i in items)
    # ...and under case folding, so case-insensitive scoring merges no gold names.
    assert all(len({entity_key(a) for a in i["answer"]}) == len(i["answer"]) for i in items)
    # Stratification metadata reaches the record.
    meta = items[0]["meta"]
    assert {"path", "target_type", "n_gold", "size_bucket"} <= set(meta)
