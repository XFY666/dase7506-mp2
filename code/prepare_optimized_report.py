"""Bind existing experiment, selection and evaluation evidence for the report.

No data prediction, training or score selection occurs here.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'runs/optimization_final_analysis'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def main():
    raw = read(OUT / 'summary.json')
    enrichment = read(OUT / 'report_analysis_enrichment.json')
    analysis = dict(raw)
    analysis.update(enrichment)
    analysis.pop('additional_evidence', None)
    analysis['status'] = 'complete'
    analysis['selected_name'] = raw['selected_name']
    analysis['validation'] = raw['validation']
    reproduced = read(ROOT / 'runs/optimized_reproduction_work/summary.json')
    analysis['search_cost_rows'][1] = dict(stage='Independent selected-recipe reproduction',
        model_fits=reproduced['model_fit_calls'],
        cost_text='10 assets / 91 arrays exactly equal; 32 fresh training-input predictions equal')
    analysis['search_cost_rows'][0]['cost_text'] = '333 + 275 + 133 research fits; 3 historical verification fits separately'
    resources = read(ROOT / 'runs/optimization_evaluation/test_resources.json')
    config = ROOT / 'runs/optimized_bundle/config.json'
    score = ROOT / 'runs/optimization_evaluation/test_score.json'
    analysis['resources'] = dict(resources, hardware=resources['local_cpu'],
        config_sha256=sha(config), score_sha256=sha(score))
    analysis['reproduction'] = {
        'python': 'Python 3.12.14; NumPy 2.5.3 inference; SciPy 1.18.1, scikit-learn 1.9.1 and CPU PyTorch 2.14.1 for refitting',
        'commands': [
            'python -m pip install -r requirements.txt',
            'python fetch_provided_data.py --splits train validation test --download',
            'python freeze_selected.py --verify',
            'python measure_selected.py --output-dir runs/reproduction_evaluation',
            'python train_optimized.py --recipe configs/selected_recipe.json --output runs/reproduced_model --work-dir runs/reproduced_work'
        ],
        'evidence': ['E_REQUIREMENTS','E_TRAIN_REQUIREMENTS','E_SEQUENCE_REQUIREMENTS','E_TRAIN_COMMAND','E_REPRODUCTION']
    }
    write(OUT / 'report_analysis.json', analysis)
    sources = {
        'E_FINAL_SELECTION': ('runs/optimization_final_selection/selection.json', 'Fixed final selection, recorded before final testing'),
        'E_FINAL_ANALYSIS': ('runs/optimization_final_analysis/report_analysis.json', 'Narrative projection of saved diagnostics and final resource evidence'),
        'E_RAW_ANALYSIS': ('runs/optimization_final_analysis/summary.json', 'Selected validation predictions, matched comparisons and illustrative cases'),
        'E_ANALYSIS_REVIEW': ('runs/optimization_final_analysis/report_analysis_enrichment.json', 'Source-checked illustrative examples and evidence-linked interpretation'),
        'E_EXAMPLE_REVIEW': ('runs/optimization_final_analysis/example_source_review.json', 'Parsed-answer and mathematical review of the two illustrative cases'),
        'E_FINAL_TEST': ('runs/optimization_evaluation/test_score.json', 'Official final frozen test score'),
        'E_FINAL_RESOURCES': ('runs/optimization_evaluation/test_resources.json', 'Complete fresh-process test evaluation time and asset inventory'),
        'E_FINAL_CONFIG': ('runs/optimized_bundle/config.json', 'Selected deployed configuration'),
        'E_REPRODUCTION': ('runs/optimization_reproduction_verification/verification.json', 'Independent full refit and exact learned-array/prediction comparison'),
        'E_REPRODUCTION_RUN': ('runs/optimized_reproduction_work/summary.json', 'Independent ten-learner refit execution record'),
        'E_FREEZE': ('runs/optimized_freeze.json', 'Complete selected deployment source and asset freeze'),
        'E_FINAL_AUDIT': ('runs/optimization_final_analysis/final_audit.json', 'Independent aggregate result and freeze chronology audit'),
        'E_SELECTION_POLICY': ('../docs/optimization_selection_policy.md', 'Selection policy fixed before new fusion and validation results'),
        'E_REQUIREMENTS': ('requirements.txt', 'Pinned inference dependency'),
        'E_TRAIN_REQUIREMENTS': ('requirements-train.txt', 'Pinned numeric training dependencies'),
        'E_SEQUENCE_REQUIREMENTS': ('requirements-sequence.txt', 'Pinned scratch sequence training dependency'),
        'E_TRAIN_COMMAND': ('train_optimized.py', 'Portable selected-recipe refitting entry point'),
        'E_HISTORICAL_SCORE': ('runs/evaluation/final-test-score.json', 'Historical frozen D score; not used for current selection'),
        'E_HISTORICAL_INVENTORY': ('runs/research_inventory.json', 'Initial research workload inventory'),
        'E_REFINEMENT_INVENTORY': ('runs/refinement_inventory.json', 'Earlier refinement workload inventory'),
        'E_GENERALIZATION_INVENTORY': ('runs/generalization_inventory.json', 'Earlier generalization workload inventory'),
        'E_OPTIMIZATION_RESULTS': ('../docs/optimization_results.md', 'Detailed current experiment ledger and disposition'),
        'E_EXTENDED_PROTOCOL': ('../docs/extended_protocol.md', 'Earlier adaptive-search protocol and context'),
        'E_PROVENANCE': ('../docs/provenance.md', 'Course materials, reused libraries and substantive assistance disclosure'),
        'E_DATA_MANIFEST': ('data/manifest.json', 'Supplied split sizes and exact data hashes')
    }
    evidence = {}
    for key, (relative, description) in sources.items():
        path = (ROOT / relative).resolve()
        evidence[key] = dict(path=Path(os.path.relpath(path, OUT)).as_posix(), sha256=sha(path), description=description)
    def ref(key, pointer=''):
        return dict(evidence=key, pointer=pointer)
    spec = dict(schema='mp2-optimization-report-v1', human_review='pending', student_identity=None,
        evidence=evidence, final=dict(name=ref('E_FINAL_SELECTION','/selected/name'),
            validation_full=ref('E_FINAL_ANALYSIS','/validation/full'),
            validation_hard=ref('E_FINAL_ANALYSIS','/validation/hard'),
            test=ref('E_FINAL_TEST'), resources=ref('E_FINAL_ANALYSIS','/resources'),
            config=ref('E_FINAL_CONFIG'), selection=ref('E_FINAL_SELECTION')),
        analysis=ref('E_FINAL_ANALYSIS'))
    write(OUT / 'report_inputs.json', spec)
    print(json.dumps(dict(status='report_inputs_prepared', evidence_files=len(evidence), selected_name=raw['selected_name'])))


if __name__ == '__main__':
    main()
