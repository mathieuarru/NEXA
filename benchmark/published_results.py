"""Results that exist only as aggregates, never as per-question records.

One source for the reports and the figures. All values are percentages on
AlzKB with GPT-4o-mini; none can enter a paired test.

ESCARGOT_PUBLISHED
    The full ESCARGOT pipeline as reported in its publication. Accuracy for
    True/False and MCQ; for open-ended, the overlap coefficient, the only
    open-ended score the publication reports.
ESCARGOT_CYPHER_ONLY
    The released ESCARGOT implementation re-run with Cypher as its only
    knowledge access (ablation), scored by benchmark/escargot_cypher_only.py
    with the same rules as every other system; values from its
    stats/summary.json, kept without intervals. MCQ 1-hop covers 497 questions:
    entry 266 (Paraparesis, Spastic) was not asked in that run.
BENCHMARK_N
    Question counts of the published benchmark files, the denominators of the
    published figures.
"""

BENCHMARK_N = {
    "True_or_False_1hop.json": 560,
    "True_or_False_2hop.json": 540,
    "MCQ_1hop.json": 498,
    "MCQ_2hop.json": 419,
    "OpenEnded_1hop.json": 508,
    "OpenEnded_2hop.json": 450,
}

ESCARGOT_PUBLISHED = {
    "accuracy": {
        "True_or_False_1hop.json": 90.9,
        "True_or_False_2hop.json": 74.1,
        "MCQ_1hop.json": 93.4,
        "MCQ_2hop.json": 89.5,
    },
    "overlap": {
        "OpenEnded_1hop.json": 88.4,
        "OpenEnded_2hop.json": 85.8,
    },
}

ESCARGOT_CYPHER_ONLY = {
    "accuracy": {
        "True_or_False_1hop.json": 60.5357,
        "True_or_False_2hop.json": 52.9630,
        "MCQ_1hop.json": 51.1066,
        "MCQ_2hop.json": 5.0119,
    },
    "exact_match": {
        "OpenEnded_1hop.json": 79.5276,
        "OpenEnded_2hop.json": 13.7778,
    },
    "precision": {
        "OpenEnded_1hop.json": 86.3589,
        "OpenEnded_2hop.json": 35.7844,
    },
    "recall": {
        "OpenEnded_1hop.json": 86.3915,
        "OpenEnded_2hop.json": 82.9388,
    },
    "f1": {
        "OpenEnded_1hop.json": 86.2459,
        "OpenEnded_2hop.json": 43.5762,
    },
    "jaccard": {
        "OpenEnded_1hop.json": 86.1817,
        "OpenEnded_2hop.json": 35.7476,
    },
    "overlap": {
        "OpenEnded_1hop.json": 86.5555,
        "OpenEnded_2hop.json": 82.9432,
    },
}
