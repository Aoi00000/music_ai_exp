"""Obtain the one pinned inference dependency; no upstream pip installation."""
import argparse,urllib.request,zipfile,shutil
from pathlib import Path
from .core import ROOT,REV,write_json

def bootstrap():
    dest=ROOT/'vendor/lv-chordia';stamp=dest/'.project_revision'
    if dest.exists():
        if stamp.exists() and stamp.read_text().strip()==REV:return
        raise FileExistsError('Existing vendor folder has a different revision: '+str(dest))
    archive=ROOT/'downloads'/f'lv-chordia-{REV[:12]}.zip';archive.parent.mkdir(parents=True,exist_ok=True)
    if not archive.exists():
        part=archive.with_suffix('.part')
        urllib.request.urlretrieve(f'https://codeload.github.com/openmirlab/lv-chordia/zip/{REV}',part)
        part.replace(archive)
    dest.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            if member.is_dir():continue
            parts=Path(member.filename).parts[1:]
            if not parts:continue
            out=(dest/Path(*parts)).resolve()
            if not out.is_relative_to(dest.resolve()):raise ValueError('Unsafe archive path')
            out.parent.mkdir(parents=True,exist_ok=True)
            with z.open(member) as source,out.open('wb') as target:shutil.copyfileobj(source,target)
    stamp.write_text(REV);write_json(ROOT/'reports/chord_model_source.json',{'lv_chordia':REV})
    print('Pinned chord inference source and public weights are ready')

def doctor():
    import torch,librosa,mir_eval
    from .model import load_original
    from .core import dictionary
    m=load_original();out=m(torch.zeros(1,32,252))
    assert [x.shape[-1] for x in out]==[85,13,4,4,3,3]
    print('PyTorch:',torch.__version__,'CUDA:',torch.cuda.is_available(),'chord candidates:',len(dictionary()))

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['bootstrap','doctor']);a=p.parse_args()
    bootstrap() if a.command=='bootstrap' else doctor()
if __name__=='__main__':main()
