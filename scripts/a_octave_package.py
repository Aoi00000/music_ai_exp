"""Export the frozen development octave filter without changing selection."""
from pathlib import Path
import json,torch
from scripts.a_candidate_data import save,sha


def main():
    source=Path('reports/octave_v1/selection.json');r=json.load(open(source));best=r['best']
    if best is None:raise ValueError('No selected octave filter')
    out=Path('models/octave_filter_review.pt')
    if out.exists():raise ValueError('Do not overwrite review weights')
    state=torch.load(best['checkpoint'],map_location='cpu',weights_only=True);state.update(threshold=best['threshold'],selection_sha256=sha(source),automatic_application_approved=False)
    torch.save(state,out);save(out.with_suffix('.json'),dict(sha256=sha(out),selection_sha256=sha(source),arm=best['arm'],step=best['step'],threshold=best['threshold'],automatic_application_approved=False));print('OCTAVE_REVIEW_PACKAGED',out)


if __name__=='__main__':main()
