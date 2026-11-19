"""Verify the handoff ZIP using only Python's standard library (no extraction)."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, zipfile

def main():
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    expected=json.loads((root/'docs/measurements/B_HANDOFF_20261119.json').read_text())
    digest=hashlib.sha256(a.archive.read_bytes()).hexdigest()
    if digest!=expected['archive_sha256']:raise ValueError('Archive SHA256 mismatch')
    prefix='band2tab_AB_integration_20261119/'
    with zipfile.ZipFile(a.archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate ZIP member')
        if z.testzip() is not None:raise ValueError('ZIP CRC mismatch')
        for name in names:
            if not name.startswith(prefix) or '..' in PurePosixPath(name).parts:raise ValueError('Unexpected ZIP path')
        sums=json.loads(z.read(prefix+'checksums.json'))
        if set(names)!={prefix+rel for rel in sums}|{prefix+'checksums.json'}:raise ValueError('Unexpected ZIP contents')
        for rel,digest in sums.items():
            if hashlib.sha256(z.read(prefix+rel)).hexdigest()!=digest:raise ValueError('Member mismatch: '+rel)
    print(json.dumps(dict(verified=True,members=len(names),bytes=a.archive.stat().st_size,archive_sha256=expected['archive_sha256'])))

if __name__=='__main__':main()
