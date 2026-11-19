"""Create a private, small optional addendum to the original A/B fixture ZIP."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    base=json.loads((ROOT/'docs/measurements/B_HANDOFF_20261119.json').read_text())
    source=ROOT/'results/b_handoff_20261119_track13/conservative_review/notes.json'
    notes=json.loads(source.read_text())
    assert notes['parts']==notes['raw_parts']
    for part in ('guitar','bass'):assert notes['suggested_parts'][part]==notes['raw_parts'][part]
    readme='''Band2Tab A/B optional review addendum, 2026-11-19
Requires the ORIGINAL band2tab_AB_integration_20261119.zip, not included here.
This package is private handoff material, not a public dataset distribution.
Load review/notes.json only after preserving the original baseline and edits.
parts/raw_parts are ORIGINAL; suggested_parts are OPTIONAL proposals.
review/suggested_notes.json exposes the proposal in parts for explicit imports.
The 12-second proposal is IDENTITY: guitar90 bass134. Not an accuracy success.
BPM, meter and downbeat remain UNKNOWN. Keep user edits and timing unchanged.
weights/: torch weights_only bundles, not a virtual environment or Mu weights.
For inference use the A branch code and the EXACT Mu medium baseline hash.
The additional route does NOT need YourMT3 weights; it does need the original
mix/guitar/bass WAVs and raw_guitar.mid/raw_bass.mid produced by A pipeline.
Do not apply to small-model, edited, differently-timed or mismatched results.
Never auto-replace baseline. Final GUI/MusicXML acceptance is still pending.
See docs/B_AGENT_UPDATE_20261119.md in Git for commands and acceptance checks.
'''
    members={'README.txt':readme.encode(),'review/notes.json':source.read_bytes(),
       'review/suggested_notes.json':json.dumps({**notes,'parts':notes['suggested_parts'],'review_explicitly_selected':True},ensure_ascii=False,indent=2).encode()}
    for name in ('bass_sound_keep_review.pt','octave_filter_review.pt'):members['weights/'+name]=(ROOT/'models'/name).read_bytes()
    manifest={'base_archive':base['archive'],'base_archive_sha256':base['archive_sha256'],
       'source_id':base['source_id'],'baseline_counts':base['baseline_counts'],'suggested_counts':{'guitar':90,'bass':134},
       'identity_on_fixture':True,'independent_test':False,'automatic_application_approved':False,
       'bpm':None,'meter':None,'first_downbeat_seconds':None,
       'muscriptor_sha256':notes['muscriptor_sha256'],'source_analysis_sha256':notes['source_analysis_sha256'],
       'model_sha256':{p:sha(b) for p,b in members.items() if p.startswith('weights/')}}
    members['model_manifest.json']=json.dumps(manifest,indent=2).encode()
    members['checksums.json']=json.dumps({p:sha(b) for p,b in members.items()},indent=2).encode()
    out=ROOT/'output/handoff/band2tab_review_addendum_20261119.zip'
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for p,b in members.items():z.writestr(p,b)
    with zipfile.ZipFile(out) as z:
        checks=json.loads(z.read('checksums.json'))
        for p,h in checks.items():assert sha(z.read(p))==h
    public={'archive':str(out.relative_to(ROOT)),'sha256':sha(out.read_bytes()),'bytes':out.stat().st_size,
       'members':len(members),'all_member_sha256_verified':True,**manifest}
    (ROOT/'docs/measurements/B_REVIEW_ADDENDUM_20261119.json').write_text(json.dumps(public,indent=2)+'\n')
    print(json.dumps(public,indent=2))
if __name__=='__main__':main()
