"""Detached local inference, no shell; write live logs and measured resource use."""
import json
import os
import platform
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def check_weights(paths):
    import hashlib
    specs={v['path']:v for v in json.loads((ROOT/'docs/measurements/B_AB_MODEL_REQUIREMENTS_20261119.json').read_text())['weights']}
    missing=[]
    for relative in paths:
        path=ROOT/relative;spec=specs[relative]
        if not path.is_file():missing.append(relative);continue
        if path.stat().st_size!=spec['bytes']:
            missing.append(relative+' (サイズ不一致)');continue
        with path.open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=spec['sha256']:missing.append(relative+' (SHA256不一致)')
    return missing


def requirements(engine,python,model):
    missing=[]
    python=Path(python)
    if not python.is_file():missing.append(str(python))
    if engine=='separated':
        missing+=check_weights(['models/muscriptor-medium/model.safetensors','models/torch/hub/checkpoints/5c90dfd2-34c22ccb.th'])
        if python.is_file():
            check=subprocess.run([str(python),'-c','import muscriptor,demucs,librosa,soundfile,imageio_ffmpeg'],capture_output=True,text=True)
            if check.returncode:missing.append('採譜環境の依存: '+check.stderr[-900:])
    elif python.is_file():
        if not (python.parent/'muscriptor').is_file():missing.append(str(python.parent/'muscriptor'))
        check=subprocess.run([str(python),'-c','import imageio_ffmpeg'],capture_output=True,text=True)
        if check.returncode:missing.append('imageio-ffmpeg')
    return missing


def launch(source,parent,engine='direct',python=None,model='small',device='cpu',seconds=12):
    import tempfile
    if engine not in ('direct','separated') or model not in ('small','medium') or device not in ('cpu','mps') or not 0<seconds<=12:
        raise ValueError('解析条件が不正です。')
    python=Path(python or ROOT/'.venv-mu-b/bin/python').absolute()
    if engine=='separated' and (model!='medium' or device!='cpu'):raise ValueError('Aの分離経路はmedium / CPUに固定されています。')
    missing=requirements(engine,python,model)
    if missing:raise ValueError('必要なファイル・依存がありません:\n'+'\n'.join(missing))
    parent=Path(parent);parent.mkdir(parents=True,exist_ok=True)
    out=Path(tempfile.mkdtemp(prefix='inference_',dir=parent))
    job=dict(source=str(Path(source).resolve()),root=str(ROOT),python=str(python),engine=engine,
             model=model,device=device,dtype='float32',seconds=seconds,status='starting',created=time.time())
    path=out/'job.json';path.write_text(json.dumps(job))
    log=(out/'worker.log').open('w')
    subprocess.Popen([sys.executable,'-m','band2tab.inference_job',str(path)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close()
    return path


def run(path):
    path=Path(path);job=json.loads(path.read_text());out=path.parent
    started=time.monotonic();job['free_before_bytes']=shutil.disk_usage(out).free
    def save(**updates):
        job.update(updates);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(job,ensure_ascii=False,indent=2));tmp.replace(path)
    env=dict(os.environ,PYTHONPATH=str(ROOT),HF_HOME=str(ROOT/'models/huggingface'),TORCH_HOME=str(ROOT/'models/torch'),PYTHONUNBUFFERED='1')
    try:
        with (out/'transcribe.log').open('w') as log:
            if job['engine'] in ('addition','chords'):
                analysis=Path(job['source']);target=out/'analysis'
                env['PYTHONPATH']=os.pathsep.join(str(ROOT/p) for p in ['vendor/yourmt3_deps','vendor/yourmt3/amt/src','vendor/yourmt3',''])
                env.update(HF_HUB_DISABLE_TELEMETRY='1',WANDB_MODE='disabled',OMP_NUM_THREADS='2')
                save(status='running',stage='補正候補を別フォルダへ作成' if job['engine']=='addition' else '同じ原音のコード認識',analysis=str(target))
                if job['engine']=='addition':
                    command=[job['python'],'-m','scripts.a_addition_apply',str(analysis),'--out',str(target)]
                else:
                    command=[job['python'],'-m','chordchart.export',str(analysis/'native.wav'),'--out',str(target)]
                subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                if job['engine']=='addition':
                    for relative in ['native.wav','chords.json','stems/guitar.wav','stems/bass.wav','stems/mix.wav']:
                        src=analysis/relative;dst=target/relative
                        if src.is_file():
                            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
            elif job['engine']=='separated':
                save(status='running',stage='分離→MuScriptor採譜',analysis=str(out/'analysis'))
                command=[job['python'],'-m','scripts.a_band_pipeline',job['source'],'--out',str(out/'analysis'),'--seconds',str(job['seconds'])]
                subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            else:
                analysis=out/'analysis';analysis.mkdir();save(status='running',stage='音声変換',analysis=str(analysis))
                code='import imageio_ffmpeg,subprocess,sys;subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-nostdin","-n","-i",sys.argv[1],"-t",sys.argv[3],"-vn","-ac","2","-ar","44100",sys.argv[2]],check=True)'
                subprocess.run([job['python'],'-c',code,job['source'],str(analysis/'native.wav'),str(job['seconds'])],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                save(stage='共通の16kHzモノラルへ変換')
                import wave,math
                import numpy as np
                from scipy.signal import resample_poly
                with wave.open(str(analysis/'native.wav'),'rb') as audio:
                    if audio.getsampwidth()!=2:raise ValueError('変換後のPCM16音声を確認できません。')
                    y=np.frombuffer(audio.readframes(audio.getnframes()),dtype='<i2').astype(np.float32)/32768
                    y=y.reshape(-1,audio.getnchannels()).mean(axis=1);sample_rate=audio.getframerate()
                    duration=audio.getnframes()/sample_rate
                divisor=math.gcd(sample_rate,16000)
                canonical=resample_poly(y,16000//divisor,sample_rate//divisor)
                write_code='import soundfile as sf,numpy as np,sys; sf.write(sys.argv[1],np.frombuffer(sys.stdin.buffer.read(),dtype="<f4"),16000,subtype="FLOAT")'
                subprocess.run([job['python'],'-c',write_code,str(analysis/'input_mu.wav')],input=canonical.astype('<f4').tobytes(),env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                save(stage='直接ミックスMuScriptor採譜')
                subprocess.run([str(Path(job['python']).parent/'muscriptor'),'transcribe',str(analysis/'input_mu.wav'),'--model',job['model'],'--device',job['device'],'--dtype','float32','--detect-tempo','false','--output',str(analysis/'raw.mid')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                from scripts.b_inspect_muscriptor_midi import remove_duplicate_metadata
                from .score import read_midi
                normalized,_=remove_duplicate_metadata((analysis/'raw.mid').read_bytes());(analysis/'normalized.mid').write_bytes(normalized)
                parts=read_midi(analysis/'normalized.mid')
                if not any(parts.values()):raise ValueError('ギター／ベース判定が0音です。元MIDIを確認してください。')
                (analysis/'notes.json').write_text(json.dumps(dict(parts=parts,raw_parts=parts,duration=duration,midi_tempo_is_placeholder=True,detected_tempo=None,corrector_applied=False)))
        if job['engine']!='chords':
            from .integration import validate_review
            result=validate_review(json.loads((Path(job['analysis'])/'notes.json').read_text()))
            job['note_counts']={p:len(ns) for p,ns in result['parts'].items()}
        else:
            result=json.loads((Path(job['analysis'])/'chords.json').read_text())
            if not isinstance(result.get('segments'),list):raise ValueError('コード区間JSONがありません。')
        version_code='import importlib.metadata as m,json; names=["muscriptor","torch","demucs","numpy","torchaudio"]; print(json.dumps({n:m.version(n) for n in names if m.packages_distributions().get(n)}))'
        versions=subprocess.run([job['python'],'-c',version_code],env=env,capture_output=True,text=True)
        try:job['package_versions']=json.loads(versions.stdout)
        except (ValueError,TypeError):job['package_versions']=dict(unavailable=True)
        revision=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
        job['git_commit']=revision.stdout.strip() if revision.returncode==0 else None
        specs=json.loads((ROOT/'docs/measurements/B_AB_MODEL_REQUIREMENTS_20261119.json').read_text())['weights']
        if job['engine']=='separated':
            job['validated_weights']=[v for v in specs if v['path'] in ('models/muscriptor-medium/model.safetensors','models/torch/hub/checkpoints/5c90dfd2-34c22ccb.th')]
        elif job['engine']=='direct':
            code='from huggingface_hub import hf_hub_download; import hashlib,json,sys,os; p=hf_hub_download("MuScriptor/muscriptor-"+sys.argv[1],"model.safetensors",local_files_only=True); f=open(p,"rb"); print(json.dumps(dict(sha256=hashlib.file_digest(f,"sha256").hexdigest(),bytes=os.path.getsize(p))))'
            meta=subprocess.run([job['python'],'-c',code,job['model']],env=env,capture_output=True,text=True)
            try:job['model_weight']=json.loads(meta.stdout)
            except (ValueError,TypeError):job['model_weight']=dict(hash_unavailable=True)
        save(status='complete',stage='結果を別フォルダへ保存完了')
    except Exception as error:
        save(status='failed',error=str(error))
    finally:
        usage=resource.getrusage(resource.RUSAGE_CHILDREN)
        save(elapsed_seconds=time.monotonic()-started,free_after_bytes=shutil.disk_usage(out).free,
             max_child_rss_bytes=int(usage.ru_maxrss)*(1 if sys.platform=='darwin' else 1024),
             memory_measurement='getrusage(RUSAGE_CHILDREN).ru_maxrss; inference child peak, excludes UI',
             host=dict(platform=platform.platform(),architecture=platform.machine(),python=sys.version.split()[0]))


def launch_aux(kind,analysis,parent,python=None):
    import tempfile
    if kind not in ('addition','chords'):raise ValueError('処理の指定が不正です。')
    analysis=Path(analysis).resolve();python=Path(python or ROOT/'.venv-yourmt3/bin/python').absolute()
    missing=[]
    if not python.is_file():missing.append(str(python))
    specs=json.loads((ROOT/'docs/measurements/B_AB_MODEL_REQUIREMENTS_20261119.json').read_text())['weights']
    paths=[v['path'] for v in specs if ('lv-chordia/weights/' in v['path'] if kind=='chords' else v['path'] in ('models/addition_jazz_v3_review.pt','models/yourmt3_lora_augmentation_review.pt') or 'checkpoints/last.ckpt' in v['path'])]
    missing+=check_weights(paths)
    needed=['native.wav'] if kind=='chords' else ['notes.json','stems/guitar.wav','stems/bass.wav','stems/mix.wav']
    missing += [str(analysis/p) for p in needed if not (analysis/p).is_file()]
    if kind=='addition' and not (ROOT/'vendor/yourmt3/amt/src').is_dir():missing.append('vendor/yourmt3/amt/src')
    if missing:raise ValueError('必要なファイルがありません。既存の結果は保持します:\n'+'\n'.join(missing))
    parent=Path(parent);parent.mkdir(parents=True,exist_ok=True)
    out=Path(tempfile.mkdtemp(prefix=kind+'_',dir=parent));path=out/'job.json'
    job=dict(source=str(analysis),python=str(python),engine=kind,model='YourMT3補完候補' if kind=='addition' else '公開5モデルコード認識',device='cpu',dtype='float32',status='starting',created=time.time())
    path.write_text(json.dumps(job));log=(out/'worker.log').open('w')
    subprocess.Popen([sys.executable,'-m','band2tab.inference_job',str(path)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
    return path


if __name__=='__main__':run(sys.argv[1])
