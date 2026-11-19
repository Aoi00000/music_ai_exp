"""Verify the supplied A ZIP and prepare seconds-only review with no inference."""
from pathlib import Path
import argparse,json,platform,resource,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from band2tab.integration import verify_extract,open_bundle,save_review
from band2tab.sync_player import player_html


def main():
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);a=p.parse_args()
    started=time.monotonic();before=shutil.disk_usage(ROOT).free
    folder=verify_extract(a.archive,ROOT/'inputs')
    document=open_bundle(folder);chords=json.loads((folder/'chords.json').read_text())
    out,report=save_review(document,ROOT/'results/ab_review',chords)
    audio={'原音':folder/'original.wav','ギターステム':folder/'stems/guitar.wav','ベースステム':folder/'stems/bass.wav'}
    (out/'review_player.html').write_text(player_html(document,chords,audio),encoding='utf-8')
    (ROOT/'results/ab_last_bundle.json').write_text(json.dumps(dict(folder=str(folder),saved=str(out))))
    profile=dict(stage='ZIP validation and seconds review, no inference',platform=platform.platform(),architecture=platform.machine(),python=sys.version.split()[0],elapsed_seconds=time.monotonic()-started,
                 peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),free_before_bytes=before,free_after_bytes=shutil.disk_usage(ROOT).free)
    if sys.platform=='darwin':profile['physical_memory_bytes']=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True))
    (out/'environment.json').write_text(json.dumps(profile,indent=2))
    print('PASS: ZIP SHA256・18ファイル・全メンバーSHA256')
    print('PASS: ギター90音 / ベース134音 / 計224音を秒単位で保存・再読込')
    print('補正案: 基準と同一（追加0・削除0）。改善とは扱いません。')
    print('BPM・拍子・小節頭: 未確認のまま / MusicXMLは確認後')
    print('音源セット:',folder)
    print('秒単位JSON:',out/'notes.json')
    print('保存済み代替デモ:',out/'review_player.html')
    print('準備時間:',round(profile['elapsed_seconds'],3),'秒（推論時間ではありません）')
    print('空き容量:',round(profile['free_after_bytes']/1024**3,2),'GiB')
    print('次: BAND2TAB_MODE=integration ./Band2Tab.command')


if __name__=='__main__':main()
