"""Package a small, hash-verified, weight-free A-to-B integration fixture.

Not an accuracy benchmark. Model inference is performed by existing commands.
No credentials, restricted GOAT recordings, or model weights enter the archive.
"""
from pathlib import Path
from collections import Counter
import argparse, hashlib, json, math, shutil, subprocess, zipfile
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def key(n):return (n['pitch'],n['onset'],n['offset'],n.get('velocity'))
def validate_notes(d,duration):
    assert set(d['parts'])=={'guitar','bass'}
    assert d['duration']==duration
    for field in ('parts','raw_parts','suggested_parts'):
        for notes in d.get(field,{}).values():
            for n in notes:
                assert isinstance(n['pitch'],int) and 0<=n['pitch']<=127
                assert math.isfinite(n['onset']) and math.isfinite(n['offset'])
                assert 0<=n['onset']<n['offset']<=duration+1e-6

def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,default=ROOT/'results/b_handoff_20261119_track13')
    p.add_argument('--out',type=Path,default=ROOT/'output/handoff');a=p.parse_args()
    source=a.analysis;name='band2tab_AB_integration_20261119';dest=a.out/name
    if dest.exists():raise ValueError('Preserve previous package; choose a new --out')
    base=json.loads((source/'baseline/notes.json').read_text());review=json.loads((source/'review/notes.json').read_text())
    chord=json.loads((source/'chords/chords.json').read_text());duration=base['duration']
    validate_notes(base,duration);validate_notes(review,duration)
    assert review['source_analysis_sha256']==sha(source/'baseline/notes.json')
    assert chord['duration']==duration
    for part in ('guitar','bass'):
        assert Counter(map(key,base['parts'][part]))==Counter(map(key,review['parts'][part]))
        assert Counter(map(key,base['parts'][part]))==Counter(map(key,review['raw_parts'][part]))
    # Copy the complete analysis files, including all original provenance fields.
    copies={'original.wav':'baseline/native.wav','baseline/notes.json':'baseline/notes.json',
        'baseline/raw_guitar.mid':'baseline/raw_guitar.mid','baseline/raw_bass.mid':'baseline/raw_bass.mid',
        'review/notes.json':'review/notes.json','chords.json':'chords/chords.json',
        'chords.csv':'chords/chords.csv','chord_chart.html':'chords/chord_chart.html',
        'stems/guitar.wav':'baseline/stems/guitar.wav','stems/bass.wav':'baseline/stems/bass.wav',
        'stems/mix.wav':'baseline/stems/mix.wav'}
    for target,origin in copies.items():
        target=dest/target;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/origin,target)
    # A convenience view for consumers which only understand `parts`.
    selected={**base,'parts':review['suggested_parts'],'raw_parts':base['parts'],
        'suggested_parts':review['suggested_parts'],'requires_review':True,'automatic_application_approved':False,
        'selected_view':'explicit_review_suggestion','baseline_file':'../baseline/notes.json',
        'review_file':'notes.json'}
    save(dest/'review/suggested_notes.json',selected)
    manifest_rows=json.loads((ROOT/'backups/azure/data/candidate_v2/manifest.json').read_text())
    row=next(r for r in manifest_rows if r['id']=='ref_mix_Track00013')
    assert sha(ROOT/'backups/azure/data/refiner_clips/ref_mix_Track00013.wav')==base['source_sha256']
    save(dest/'reference/notes.json',dict(parts=row['parts'],duration=duration,
        role='Dataset reference for debugging only, never input to inference. Training fixture, not held-out evaluation.',
        labels='Existing project sounding-pitch labels, including documented Slakh bass-octave correction and boundary censoring.'))
    audio={}
    for rel in ('original.wav','stems/mix.wav','stems/guitar.wav','stems/bass.wav'):
        info=sf.info(dest/rel);assert abs(info.duration-duration)<1/info.samplerate+1e-8
        audio[rel]=dict(frames=info.frames,sample_rate=info.samplerate,channels=info.channels,duration=info.duration)
    # The inference pipeline did no cropping or delay compensation per stem.
    # Check its packaged canonical mix against the original conversion exactly.
    from band2tab.separation import resample
    y,sr=sf.read(dest/'original.wav',dtype='float32',always_2d=True)
    expected=resample(y.mean(1),sr,16000)
    mix,mr=sf.read(dest/'stems/mix.wav',dtype='float32')
    assert mr==16000 and len(expected)==len(mix) and np.array_equal(expected,mix)
    baseline=base['parts'];suggested=review['suggested_parts'];diff={}
    for part in ('guitar','bass'):
        old,new=Counter(map(key,baseline[part])),Counter(map(key,suggested[part]))
        diff[part]=dict(baseline=len(baseline[part]),suggested=len(suggested[part]),
            removed=list((old-new).elements()),added=list((new-old).elements()))
    # Record all weights actually needed, but do not put them in the small ZIP.
    weight_paths=[ROOT/'models/muscriptor-medium/model.safetensors',
        ROOT/'models/torch/hub/checkpoints/5c90dfd2-34c22ccb.th',
        ROOT/'models/yourmt3_lora_augmentation_review.pt',ROOT/'models/addition_jazz_v3_review.pt',
        ROOT/'models/octave_filter_review.pt']
    from scripts.a_yourmt3_fetch import CHECKPOINT
    weight_paths += [ROOT/'vendor/yourmt3'/CHECKPOINT]
    from chordchart.model import original_path
    weight_paths += [original_path(i) for i in range(5)]
    weights=[dict(path=str(w.relative_to(ROOT)),bytes=w.stat().st_size,sha256=sha(w)) for w in weight_paths]
    assert weights[0]['sha256']==base['muscriptor_sha256']
    assert weights[2]['sha256']==review['proposal_adapter_sha256']
    assert weights[3]['sha256']==review['ranker_sha256']
    assert weights[4]['sha256']==review['octave_filter_sha256']
    save(dest/'model_manifest.json',dict(weights_included=False,weights=weights,
        models=dict(separator='htdemucs_6s / Demucs 4.0.1',transcriber='MuScriptor medium / muscriptor 0.4.0',
            proposal='YourMT3+ with augmentation LoRA; frozen',addition='addition_jazz_v3_review; threshold in checkpoint',
            octave='octave_filter_review; threshold in checkpoint',chords='published 5-fold CRNN ensemble; no project chord-head checkpoint'),
        configuration=dict(device='cpu',dtype='float32',mu_native_instrument_conditioning=False,
            role_policy='Every non-drum event from the estimated G/B stem assigned to that source',
            demucs_seed=int(sha(source/'baseline/native.wav')[:8],16)%2**31,demucs_shifts=1,demucs_overlap=.25,demucs_segment_seconds=7,
            note_time_unit='seconds',crop_or_time_shift_after_input=False,tempo_detection_in_mu=False,
            app_auto_replacement=False),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        pipeline_sha256=sha(ROOT/'scripts/a_band_pipeline.py'),addition_script_sha256=sha(ROOT/'scripts/a_addition_apply.py')))
    save(dest/'timing.json',dict(duration_seconds=duration,time_origin='original.wav sample 0 = every JSON and every stem 0 seconds',
        source_excerpt_start_seconds=row['excerpt_start'],source_excerpt_start_applied_before_input=True,
        bpm=dict(value=None,status='unconfirmed'),meter=dict(numerator=None,denominator=None,status='unconfirmed'),
        first_downbeat_seconds=dict(value=None,status='unconfirmed'),
        estimated_bpm_from_chord_pipeline=chord.get('tempo'),
        midi_tempo_120_is_placeholder=True,instructions='Do not add the dataset excerpt offset to packaged JSON times. Keep seconds until tempo and downbeat are confirmed by the user.'))
    save(dest/'verification.json',dict(schema='band2tab_AB_fixture_v1',fixture_only=True,natural_band_recording=False,
        source_dataset='BabySlakh/Slakh, ref_mix_Track00013; already-used training excerpt',
        duration_seconds=duration,audio=audio,same_time_origin_by_construction=True,
        canonical_mix_matches_original=True,baseline_complete=True,baseline_matches_review_raw=True,
        note_counts_and_diff=diff,proposal_candidates=review['proposal_count'],
        correction_is_identity=all(not v['added'] and not v['removed'] for v in diff.values()),
        timings_on_A_cpu_seconds=dict(separation_and_mu=base['seconds'],your_proposal_and_review=review['seconds']),
        timing_scope='One A Mac run only; no B 16GiB small/medium speed or memory claim',
        validation_not_done=['B GUI import/edit/save','playback-to-TAB synchronization','B Mac runtime/memory','natural band accuracy','tempo/meter/downbeat human confirmation']))
    readme='''# A→B 小規模結合確認セット（2026-11-19）

12秒の同じ入力に対する、完全な音符・補正案・コード結果です。
BabySlakh/Slakhの合成バンドを使う操作確認用で、未知曲の精度評価ではありません。
最初に採用したTrack00020はギター検出0音だったため失敗例としてA側に保管し、
このセットには両パートが検出されたTrack00013を採用しました。成功率の主張ではありません。

## 最初に開くファイル

- original.wav：12秒の入力原音。G/B以外を含む合成バンド。
- stems/guitar.wav / bass.wav：推定分離音。漏れ・欠落はあり得ます。
- baseline/notes.json：Mu medium基準。partsとraw_partsにギター90音・ベース134音。
- review/notes.json：parts/raw_partsは基準、suggested_partsが学習済み補正案。
- review/suggested_notes.json：partsだけを読む画面向けの明示的な補正案表示用。
- chords.json：同じ原音の標準5モデルによるコード区間。chords.csvと印刷HTMLも同梱。
- timing.json：拍情報。BPM・拍子・最初の小節頭はすべて未確認（null）。
- model_manifest.json：実際のモデル・設定・重みSHA256。重み自体は同梱しません。
- reference/notes.json：既存データの参照音符。推論には使っていません。編集箇所の候補確認用。
- verification.json / checksums.json：Aで行った整合性確認と全ファイルのハッシュ。

## 時刻と補正案

0秒は常にoriginal.wavの先頭です。分離音・音符・コードで同じ原点です。
元データセットの切り出し開始をJSONへ再加算しないでください。
保持判定は onset <= t < offset。音高はMIDI番号、時刻は秒です。

今回の補正案は66候補から追加0・削除0で、基準と同一です。効果を見せるために
架空の変更を作らないでください。読み込みと原結果保護を確認し、人の修正を別保存してください。
このセットだけで補正の精度改善は説明できません。実験の改善は発表PDFの別データで示します。

## Bの受入確認（これから行うこと）

1. checksums.jsonを照合。原音と両分離音の長さが12秒であることを確認。
2. 基準と補正案を別の状態として読み込む。全音一覧にギターとベースの両方を残す。
3. 原音・分離音を聞き、参照ラベルも補助にして誤りを1か所だけ修正。元ファイルは不変。
4. 他パート・未編集音・拍設定が保存前後で変わらないことを機械的に比較し、再読込。
5. コード名と実音を併記。コード外音を捨てない。拍は未確認のまま秒表示から始める。
6. BPM等を人が確認できた場合だけ譜面設定を確定し、MusicXML保存まで確認。
7. その後にAのローカル認識処理を画面から呼ぶ。まず10〜12秒、CPU/mediumとsmallを区別。

## 実機・権利・共有

音源の出典：Slakh/BabySlakh, Manilow et al., Learning to Separate Instruments from
Synthesized Audio (WASPAA 2019), https://www.slakh.com/ 。元データの条件・帰属表示に従ってください。
研究用の内部結合確認用です。GOAT制限音源、認証、重みを含みません。
このZIPをGitにコミットしないでください。モデルを導入する際は各モデル条件を別途確認。
Aの観測時間はverification.jsonにあります。Bの16GiB Macでの速度・メモリ保証ではありません。
'''
    (dest/'README_JA.md').write_text(readme,encoding='utf-8')
    sums={str(f.relative_to(dest)):sha(f) for f in sorted(dest.rglob('*')) if f.is_file()}
    save(dest/'checksums.json',sums)
    archive=a.out/(name+'.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(dest.rglob('*')):
            if f.is_file():z.write(f,str(Path(name)/f.relative_to(dest)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for rel,digest in sums.items():assert hashlib.sha256(z.read(name+'/'+rel)).hexdigest()==digest
    audit=dict(archive=str(archive.relative_to(ROOT)),archive_sha256=sha(archive),bytes=archive.stat().st_size,
        members=len(sums)+1,all_member_sha256_verified=True,baseline_counts={p:len(ns) for p,ns in baseline.items()},
        corrected_counts={p:len(ns) for p,ns in suggested.items()},
        correction_is_identity=all(not v['added'] and not v['removed'] for v in diff.values()),
        generated='2026-11-19',source_id=row['id'],integration_fixture_not_accuracy_evaluation=True)
    save(ROOT/'docs/measurements/B_HANDOFF_20261119.json',audit)
    print(json.dumps(audit,ensure_ascii=False))

if __name__=='__main__':main()
