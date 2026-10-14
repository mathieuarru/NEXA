"""Ablation switches (NEXA_ABLATE_* in nexa/config/params.py) and ablation sampling.

Each switch changes exactly one step and nothing when off; samples are
reproducible, proportional and nested.
"""

from collections import Counter

import pytest
from langgraph.graph import END

from benchmark import ablation_direct as ad
from nexa.config import params
from nexa.knowledge_graph import property_processing_subgraph as pps
from nexa.knowledge_graph import query_correction as qc


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------

def test_allocate_is_proportional_and_exact():
    counts = {"a": 3029, "b": 949, "c": 36, "d": 5}
    alloc = ad.allocate(counts, 1000)
    assert sum(alloc.values()) == 1000
    total = sum(counts.values())
    for k, c in counts.items():
        assert abs(alloc[k] - 1000 * c / total) < 1          # largest remainder: within one draw of the quota


def test_allocate_respects_caps_for_a_nested_sample():
    counts = Counter(("p", i % 7) for i in range(700)) + Counter({("q", i): 3 for i in range(30)})
    parent = ad.allocate(counts, 200)
    child = ad.allocate(counts, 120, caps=parent)
    assert sum(child.values()) == 120
    assert all(child[k] <= parent[k] for k in counts)


def test_draw_is_deterministic_and_nested():
    keys = [i % 5 for i in range(500)]
    perms = ad.stratum_permutations(keys, seed=7)
    assert perms == ad.stratum_permutations(keys, seed=7)
    big = ad.draw(perms, ad.allocate(Counter(keys), 100))
    small = ad.draw(perms, ad.allocate(Counter(keys), 40, caps=ad.allocate(Counter(keys), 100)))
    assert set(small) <= set(big) and len(small) == 40
    assert Counter(keys[i] for i in big) == Counter({k: 20 for k in range(5)})


def test_source_set_strips_sample_suffixes():
    assert ad.source_set("MCQ_1hop_abl.json") == "MCQ_1hop.json"
    assert ad.source_set("BioHopR_2hop_abl1000.json") == "BioHopR_2hop.json"
    assert ad.source_set("OpenEnded_1hop_perturbed.json") == "OpenEnded_1hop.json"
    assert ad.source_set("True_or_False_2hop_atleastone.json") == "True_or_False_2hop_atleastone.json"


# ---------------------------------------------------------------------------
# perturbation set
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question, kind, name", [
    ("List the genes which bind to the drug Stanozolol", "drug", "Stanozolol"),
    ("Which of the following classes does not include the drug Ivacaftor? 1. A 2. B", "drug", "Ivacaftor"),
    ("Which of the following increases the gene expression of FOXM1? 1. A", "gene", "FOXM1"),
    ("List the drugs which belong to the drug class Antiemetics", "drug class", "Antiemetics"),
])
def test_find_anchor(question, kind, name):
    k, n, start, end = ad.find_anchor(question)
    assert (k, n) == (kind, name) and question[start:end] == name


def test_find_anchor_none_without_anchor_phrase():
    assert ad.find_anchor("List the symptoms of Alzheimer's Disease") is None


def test_perturbations():
    assert ad.perturb_case("Stanozolol") == "stanozolol"
    assert ad.perturb_case("adrenal gland") == "Adrenal Gland"
    typo = ad.perturb_typo("Stanozolol", "key")
    assert typo == ad.perturb_typo("Stanozolol", "key")                   # seeded
    assert len(typo) == len("Stanozolol") - 1 and typo[0] == "S" and typo[-1] == "l"
    assert ad.perturb_typo("APOE", "key") is None                          # too short to stay recognisable


def test_locate_start_finds_the_entity_that_ends_a_biohopr_question():
    q = "Name all diseases that are related to a effect/phenotype that is associated with drug Benzyl benzoate."
    start, end = ad.locate_start(q, "Benzyl benzoate")
    assert q[start:end] == "Benzyl benzoate"
    assert q[:start] + "benzyl benzoate" + q[end:] == q.replace("Benzyl benzoate", "benzyl benzoate")
    assert ad.locate_start(q, "zoate") is None                             # part of a word, not a name
    assert ad.locate_start(q, "Aspirin") is None


def test_both_benchmarks_get_the_same_experiments():
    assert ad.EXPERIMENTS["biohopr"].arms == ad.EXPERIMENTS["alzkb"].arms == list(ad.ARMS)
    assert ad.EXPERIMENTS["biohopr-perturbation"].arms == ad.EXPERIMENTS["alzkb-perturbation"].arms


def test_pooled_score_uses_each_question_primary_metric():
    tf = ("True_or_False_1hop.json", "q1")
    oe = ("OpenEnded_1hop.json", "q2")
    assert ad.primary_score(tf, {"correct": True}) == 1.0
    assert ad.primary_score(oe, {"correct": False, "metrics": {"jaccard": 0.5}}) == 0.5


# ---------------------------------------------------------------------------
# Direct switches
# ---------------------------------------------------------------------------

def _state(data, notifications=None):
    return {"data": data, "notifications": notifications,
            "attempts_info": {"global_": {"current": 0, "max": 3}, "neo4j_error": {"current": 0, "max": 3}}}


def test_empty_result_loops_by_default(monkeypatch):
    for flag in ad.ABLATION_FLAGS:
        monkeypatch.setattr(params, flag, False)
    assert qc.check_attempts(_state([])).goto == "extract_properties"
    assert qc.check_attempts(_state([], [{"code": "Neo.ClientError", "description": "x"}])).goto == "correct_neo4j_error"


def test_no_empty_loop_ends_on_empty_result(monkeypatch):
    monkeypatch.setattr(params, "NEXA_ABLATE_EMPTY_RESULT_LOOP", True)
    cmd = qc.check_attempts(_state([]))
    assert cmd.goto == END and cmd.graph == qc.Command.PARENT


def test_no_syntax_handles_an_error_like_an_empty_result(monkeypatch):
    monkeypatch.setattr(params, "NEXA_ABLATE_SYNTAX_CORRECTION", True)
    monkeypatch.setattr(params, "NEXA_ABLATE_EMPTY_RESULT_LOOP", False)
    err = [{"code": "Neo.ClientError.Statement.SyntaxError", "description": "bad"}]
    assert qc.check_attempts(_state(None, err)).goto == "extract_properties"
    # both removed ("bare"): the failing query is final
    monkeypatch.setattr(params, "NEXA_ABLATE_EMPTY_RESULT_LOOP", True)
    assert qc.check_attempts(_state(None, err)).goto == END


def test_rows_end_the_loop_whatever_the_switches(monkeypatch):
    for flag in ad.ABLATION_FLAGS:
        monkeypatch.setattr(params, flag, True)
    assert qc.check_attempts(_state([{"n": 1}])).goto == END


def test_no_db_matching_skips_the_property_subgraphs(monkeypatch):
    state = {"parsed_properties": {"key": {"suggested_values": [{"APOE": []}]}}}
    monkeypatch.setattr(params, "NEXA_ABLATE_DB_MATCHING", False)
    assert isinstance(qc.start_property_subgraphs(state), list)          # one Send per property
    monkeypatch.setattr(params, "NEXA_ABLATE_DB_MATCHING", True)
    assert qc.start_property_subgraphs(state) == "collect_all_properties"


def test_no_llm_selection_takes_the_top_candidate_without_calling_the_llm(monkeypatch):
    monkeypatch.setattr(params, "NEXA_ABLATE_LLM_SELECTION", True)
    key = "('node', 'Gene', 'geneSymbol', 'apoe')"
    state = {"property_key": key,
             "parsed_properties": {key: {"suggested_values": [{"apoe": ["APOE", "APOE4", "APOC1"]}]}}}
    out = pps.suggest_property_subgraph(state, llm=None)                 # llm=None: any call would raise
    assert out["parsed_properties"][key]["suggested_values"][-1]["selectionned_value"] == "APOE"
    # and no synonym round afterwards, even when nothing matched
    nothing = {"property_key": key,
               "parsed_properties": {key: {"suggested_values": [{"": [], "needs_synonym": True}]}}}
    assert pps.check_completion_subgraph(nothing) == END


# ---------------------------------------------------------------------------
# few-shot examples and run planning
# ---------------------------------------------------------------------------

def test_no_examples_empties_every_few_shot_block(monkeypatch):
    from nexa.models import prompts
    monkeypatch.setattr(params, "NEXA_ABLATE_FEWSHOT_EXAMPLES", False)
    assert prompts._examples_block("", "DEFAULT") == "DEFAULT"          # generic placeholders when none cached
    assert prompts._examples_block("GROUNDED", "DEFAULT") == "GROUNDED"
    monkeypatch.setattr(params, "NEXA_ABLATE_FEWSHOT_EXAMPLES", True)
    assert prompts._examples_block("GROUNDED", "DEFAULT") == ""
    system, _ = prompts.get_query_generation_prompt("q", nodes={}, relationships=[], rel_properties={},
                                                    examples="GROUNDED EXAMPLE")
    assert "GROUNDED EXAMPLE" not in system and "Example 1" not in system


def test_planned_runs_set_every_switch_explicitly():
    runs = ad.planned_runs(["alzkb"], ["full", "no-examples", "generic-examples"])
    assert len(runs) == 3 * len(ad.EXPERIMENTS["alzkb"].files)
    for _, arm, env, argv in runs:
        assert set(ad.ABLATION_FLAGS) <= set(env)                      # none left to the local .env
        on = {f for f in ad.ABLATION_FLAGS if env[f] == "true"}
        assert on == set(ad.ARMS[arm].flags)
        assert argv[argv.index("--examples-mode") + 1] == ad.ARMS[arm].examples_mode
        assert argv[argv.index("--out-subdir") + 1].endswith(ad.arm_dir(arm))


def test_example_literals_are_the_cypher_values_of_the_cached_examples():
    lits = ad.example_literals("Alzkb 121")
    assert {"Alzheimer's Disease", "Familial Alzheimer Disease (FAD)", "TBP"} <= lits
    assert not any(len(v) < 3 or v.lower() == "and" or "commonName" in v for v in lits)
    assert ad.cites_any("Which genes bind TBP?", {"TBP"}) and not ad.cites_any("Which genes bind TBPL1?", {"TBP"})
