"""Paired, source-grouped review of completed Mac evaluations.

This reads existing predictions; it never trains or opens a new test split.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from scripts.a_candidate_data import save, sha
from scripts.a_candidate_review import errors, matching
from scripts.a_candidate_train import KEYS


def missed_symptoms(row, parts):
    start=row['guard_seconds'];end=row['duration']-start if start else float('inf');result={}
    for part in ('guitar','bass'):
        refs=[n for n in row['parts'].get(part,[]) if start<=n['onset']<end and not n.get('censored_onset')]
        est=[n for n in parts.get(part,[]) if start<=n['onset']<end]
        opposite=parts.get('bass' if part=='guitar' else 'guitar',[])
        matched={i for i,j in matching(refs,est)};missed=[n for i,n in enumerate(refs) if i not in matched]
        result[part]={
            'missed_with_octave_candidate':sum(any(abs(n['pitch']-m['pitch']) in (12,24) and abs(n['onset']-m['onset'])<=.05 for m in est) for n in missed),
            'missed_with_nearby_same_pitch_attack':sum(any(n['pitch']==m['pitch'] and .05<abs(n['onset']-m['onset'])<=.2 for m in est) for n in missed),
            'missed_with_other_instrument_candidate':sum(any(n['pitch']==m['pitch'] and abs(n['onset']-m['onset'])<=.05 for m in opposite) for n in missed)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--learned', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    directory = Path('reports/yourmt3/mac_validation')
    reports = [json.loads((directory / tag / 'comparison.json').read_text())
               for tag in (args.base, args.learned)]
    base, learned = reports
    for key in ('case_ids', 'split', 'manifest_sha256', 'goat_manifest_sha256', 'device', 'dtype'):
        if base[key] != learned[key]:
            raise ValueError('Non-comparable evaluations: ' + key)
    if base['adapter_sha256'] is not None or learned['adapter_sha256'] is None:
        raise ValueError('Require raw base versus one learned adapter')
    if base['base']['checkpoint_sha256'] != learned['base']['checkpoint_sha256']:
        raise ValueError('Different pretrained bases')
    root = Path('backups/azure')
    rows = {r['id']: r for path in ('data/candidate_v2/manifest.json', 'data/goat_v1/manifest.json')
            for r in json.loads((root / path).read_text())}
    if any(rows[i]['split'] != base['split'] for i in base['case_ids']):
        raise ValueError('Manifest split mismatch')
    records = {}
    for tag in (args.base, args.learned):
        records[tag] = {i: json.loads((directory / tag / 'midi' / (i + '.json')).read_text())
                        for i in base['case_ids']}
    rng = np.random.default_rng(20261118)
    datasets = {}
    for dataset in sorted(base['summary']['datasets']):
        ids = [i for i in base['case_ids'] if rows[i]['dataset'] == dataset]
        groups = sorted({rows[i]['group'] for i in ids})
        delta = {key: learned['summary']['datasets'][dataset][key] -
                 base['summary']['datasets'][dataset][key] for key in KEYS}
        # Resample entire source groups, retaining all tone/clip conditions in a group.
        draws = rng.integers(0, len(groups), size=(2000, len(groups)))
        intervals = {}
        for key in KEYS:
            grouped = [[records[args.learned][i]['metrics'][key] - records[args.base][i]['metrics'][key]
                        for i in ids if rows[i]['group'] == group] for group in groups]
            sums = np.array([sum(v) for v in grouped]); counts = np.array([len(v) for v in grouped])
            samples = sums[draws].sum(1) / counts[draws].sum(1)
            intervals[key] = np.quantile(samples, [.025, .975]).tolist()
        diagnostics = {};symptoms={}
        for tag in (args.base, args.learned):
            cases = [errors(rows[i], records[tag][i]['parts']) for i in ids]
            diagnostics[tag] = {part: {key: sum(c[part][key] for c in cases)
                                      for key in cases[0][part]} for part in ('guitar', 'bass')}
            examples=[missed_symptoms(rows[i],records[tag][i]['parts']) for i in ids]
            symptoms[tag]={part:{key:sum(c[part][key] for c in examples) for key in examples[0][part]} for part in ('guitar','bass')}
        datasets[dataset] = {'conditions': len(ids), 'source_groups': len(groups),
                             'delta': delta, 'exploratory_paired_group_bootstrap_95': intervals,
                             'errors': diagnostics,'missed_note_symptoms':symptoms}
    result = {'split': base['split'], 'base_tag': args.base, 'learned_tag': args.learned,
              'conditions': len(base['case_ids']), 'datasets': datasets,
              'base_summary': base['summary'], 'learned_summary': learned['summary'],
              'summary_delta': {k: learned['summary'][k] - base['summary'][k] for k in KEYS},
              'report_sha256': {tag: sha(directory / tag / 'comparison.json')
                                for tag in (args.base, args.learned)},
              'bootstrap_seed': 20261118, 'bootstrap_replicates': 2000,
              'limitations': ['Validation intervals are descriptive after repeated model selection, not confirmatory significance.',
                              'Within-dataset source groups may share performers or compositions; these intervals do not remove that dependence.',
                              'Pretrained-model training overlap is not excluded.',
                              'Octave/timing/instrument symptoms are overlapping candidate explanations, not proven causes or exclusive categories.',
                              'Note/frame F1 do not measure chord naming, TAB fingering, readability or natural-band usability.']}
    save(args.out, result)
    print('PAIRED_REVIEW', args.out, result['summary_delta'], flush=True)


if __name__ == '__main__':
    main()
