"""Publish aggregate evidence without audio, credentials or per-song labels."""
from pathlib import Path
import json,argparse
from datetime import datetime
from scripts.a_candidate_data import save,sha

KEYS=('note_f1','frame_f1','active_exact_match','offset_f1')


def main():
    p=argparse.ArgumentParser();p.add_argument('--allow-pending',action='store_true');a=p.parse_args()
    sources={};data={}
    def read(name,path):
        obj=json.load(open(path));sources[name]=dict(path=path,sha256=sha(path));return obj
    data['conditioning']=read('conditioning','reports/mu_conditioning_v1/comparison.json')['summary']
    c=json.load(open('reports/mu_conditioning_v1/comparison.json'));data['polyphony']=c['polyphony']
    data['jazz_mu']=read('jazz_mu','reports/jazz_guitar_v1/mu_comparison.json')['summary']
    for name in ('addition_v1','addition_balanced_v2','addition_jazz_v3_control','addition_jazz_v3'):
        r=read(name,f'reports/{name}/selection.json');b=r['best']
        item=dict(baseline=r['control']['summary'],unfiltered_union=r['unfiltered_union']['summary'],selected=None if b is None else dict(kind=b['kind'],checkpoint=b['checkpoint'],threshold=b['threshold'],summary=b['summary']),identity_selected=r['identity_selected'])
        if name.startswith('addition_jazz_'):
            item['fixed_step250_threshold05']=next(x['summary'] for x in r['choices'] if x['checkpoint'].endswith('step_250.pt') and x['threshold']==.5)
        data[name]=item
    for name in ('addition_regression_v1','addition_jazz_regression_v1'):
        data[name]=read(name,f'reports/{name}/comparison.json')['summary']
    release=read('release','reports/release_v2/selection.json');data['release']=dict(selected=release['selected'])
    gate=read('release_gate','reports/release_gate_v3/selection.json');data['release_gate']=dict(identity_selected=gate['identity_selected'],compared_policies=len(gate['choices']))
    taxonomy=read('error_taxonomy','reports/error_taxonomy_v1/diagnostic.json');data['error_taxonomy']={k:v for k,v in taxonomy.items() if k!='tracks'}
    octave=read('octave','reports/octave_v1/selection.json');data['octave']=dict(baseline=octave['baseline']['summary'],best={k:v for k,v in octave['best'].items() if k!='tracks'},identity_selected=octave['identity_selected'])
    data['octave_regression']=read('octave_regression','reports/octave_regression_v1/comparison.json')['summary']
    chord=read('chord_head','reports/chord_head_v2/selection.json');data['chord_head']=dict(controls=chord['controls'],best={k:v for k,v in chord['best'].items() if k not in ('tracks','groups')} if chord['best'] else None,training_recordings=chord['training_recordings'],validation_recordings=chord['validation_recordings'],
        choices=[{k:v for k,v in c.items() if k not in ('tracks','groups')} for c in chord['choices']])
    data['chord_vocabulary']=read('chord_vocabulary','reports/chord_head_v2/vocabulary_diagnostic.json')
    tone=Path('backups/azure/reports/tone_v1/comparison.json')
    if tone.exists():
        r=read('tone',str(tone));data['tone']=dict(results=r['results'],matched_source_crop_and_tokens=r['matched_source_crop_and_tokens'],cases=r['cases'],device=r['device'])
        data['tone']['effective_updates']={}
        for tag in ('control','colour'):
            folder=Path('backups/azure/runs/yourmt3_lora')/f'tone_{tag}_r8_600';h=json.load(open(folder/'history.json'));data['tone']['effective_updates'][tag]=h[-1]['optimizer_updates']
    elif not a.allow_pending:raise ValueError('Do not publish final report while Azure comparison is pending')
    cleanup=Path('reports/azure_session_cleanup.json');data['azure_cleanup']=json.load(open(cleanup))
    if not a.allow_pending and not data['azure_cleanup'].get('deallocated'):raise ValueError('Azure must be deallocated before final report')
    if data['azure_cleanup'].get('deallocated'):
        proof=read('azure_deallocation','reports/azure_tone_deallocated_20261119.json');audit=read('azure_backup_audit','backups/azure/reports/candidate_v2/mac_audit.json');checkpoints=json.load(open('backups/azure/reports/backup_verification.json'))
        data['azure_cleanup'].update(observed_at=proof['observed_at'],power_state=proof['power_state'],verified_files=audit['verified_files'],verified_bytes=audit['verified_bytes'],verified_checkpoints=len(checkpoints['files']))
    native_paths=[Path('reports/idmt_bass_lines')/('tone_'+t+'_regression/comparison.json') for t in ('control','colour')]
    if all(p.exists() for p in native_paths):
        from band2tab.score import read_midi
        from scripts.a_candidate_train import aggregate,metrics
        rows=json.load(open('reports/idmt_bass_lines/manifest.json'));data['native_bass_tone']={}
        for tag in ('augmentation_regression','tone_control_regression','tone_colour_regression'):
            record=read('native_'+tag,'reports/idmt_bass_lines/'+tag+'/comparison.json');scores=[]
            for row in rows:scores.append((row,metrics(row,read_midi(Path('reports/idmt_bass_lines')/tag/'midi'/(row['id']+'.mid'),force_part='bass'))))
            data['native_bass_tone'][tag]=dict(strict=record['summary'],source_known=aggregate(scores))
        mu=read('native_mu_bass','reports/idmt_bass_lines/mu_control/comparison.json');data['native_bass_tone']['mu_control']=mu['summary']
    elif not a.allow_pending:raise ValueError('Wait for both fixed native bass regression runs')
    runs=[]
    for file in sorted(Path('backups/azure/runs/yourmt3_lora').glob('*/completed.json')):
        r=json.load(open(file));config=r['config'];best=r['best']
        runs.append(dict(name=file.parent.name,selected_step=best['step'],scope=config.get('scope'),targets=config.get('targets'),trainable_parameters=config.get('trainable_parameters'),frozen_base_unchanged=r['frozen_base_parameters_unchanged']))
    data['lora_inventory']=runs;data['sources']=sources;data['scope']='Development and previously inspected regression data; no new final held-out Slakh test opened.'
    save('docs/measurements/review_20261119.json',data)
    lines=['# 最新実験結果：2026年11月19日','', '数値は各実験内で同じ条件を比べる。別データの表同士を精度曲線として結ばない。F1は0〜1で高いほどよい。今回の正式な追加学習は17条件。別に前処理の違いで主比較から除外したコード学習2条件も記録する。推論設定の比較を新しい学習モデル数に混ぜない。', '']
    def table(rows):
        lines.append('|方式|音符F1|Frame F1|全音一致|終了F1|');lines.append('|---|---:|---:|---:|---:|')
        for label,s in rows:lines.append('|'+label+'|'+'|'.join(f'{s[k]:.4f}' for k in KEYS)+'|')
        lines.append('')
    lines+=['## 01 楽器指定は万能ではなかった','', '推定G/Bステム、同じMu重み、同じ既知ソース割当、16開発条件。推論設定の比較であり、独自の学習ではない。','']
    table([(k,v) for k,v in data['conditioning'].items()])
    lines+=['楽器を指定しても誤検出は増え得る。cfg=1/2を全入力へ自動採用しない。公式に用意された設定も、我々の入力で測定して選ぶ。','', '正解の同時音数別の全フレーム合算（曲別平均とは別集計）：','', '|正解同時音数|フレーム数|TP|FP|FN|Frame F1|','|---|---:|---:|---:|---:|---:|']
    for k,v in data['polyphony']['unconditioned'].items():lines.append(f"|{k}|{v['frames']}|{v['tp']}|{v['fp']}|{v['fn']}|{v['frame_f1']:.4f}|")
    lines+=['','one区間の低スコアは、単音自体が難しいという断定ではなく、正解が一音のとき余分な持続音等が多いことを示す。4音以上にも取りこぼしがある。全128音高・楽器を数える診断で、主要表の23〜88音高のFrame F1とは母集団が違う。コード名やジャズジャンルの分類ではない。','']
    lines+=['## 02 終了補正：採用しなかった理由','','開始・音高・楽器・音符数を固定した線形／MLP×2教材の4学習条件。最良Frameは0.8069→0.8091だが全音一致0.4208→0.4154。事前基準に届かず補正なしを選択。既存の真偽判定で対象を絞る16設定も不採用。','', '## 03 未検出ギター音を足す学習','','16開発条件。Muの全音符を保存し、YourMT3の候補を追加する。線形359パラメータとMLPを比較し、正例重み2→8を別実験で検証。','']
    first=data['addition_v1'];balanced=data['addition_balanced_v2']
    table([('Mu基準',first['baseline']),('全候補の単純追加',first['unfiltered_union']),('学習選別・重み2',first['selected']['summary']),('学習選別・重み8',balanced['selected']['summary'])])
    lines+=['重み2は1音追加だけの改善。重み8も小幅で、実用完成や全般的優位とは言えない。不要な候補を全部追加すると悪化することを対照で確認した。','', '14既知バンド条件で重み・閾値を固定した回帰確認：','']
    table(list(data['addition_regression_v1'].items()))
    lines+=['既開封の7演奏×2干渉条件であり、新しい最終テストではない。音符・Frame・全音一致はわずかに改善、終了込みは低下。','', '## 04 教材を弱点へ寄せる小規模比較','','ジャズは診断例。取得済みGuitarSetの12訓練録音を追加し、Jazz3の6録音を補完器の検証に使用した。奏者は重複。過去のモデル学習・上流事前学習まで未知という意味ではない。','']
    cj=data['addition_jazz_v3_control'];j=data['addition_jazz_v3']
    table([('Mu・指定なし',data['jazz_mu']['val_unconditioned']),('Mu・ギター指定',data['jazz_mu']['val_conditioned']),('補完学習・教材なし',cj['selected']['summary']['datasets']['jazz_guitarset']),('補完学習・12録音追加',j['selected']['summary']['datasets']['jazz_guitarset'])])
    lines+=['追加教材は12録音から94候補・43正例を補充。候補のある訓練レコードは79→91、候補は2024→2118、正例109→152。両条件1000更新・seed19・同じ標準化・線形359パラメータ。追加側は20%を新教材からサンプリングするため、教材＋配分の一式の効果である。','', '両者を同じ250更新・閾値0.5に揃えても、追加なしのジャズ音符0.6338に対し追加あり0.6920、Frame0.6955に対し0.7233。閾値選択の違いだけによる改善ではない。','', 'ただし、同じ曲群の開発検証6件であり、未知ジャンルや自然なバンドで既存モデルを上回った証明ではない。別の既知バンド14条件では両補完器が同じ結果だった。追加教材の効果は現在この診断範囲に限られる。ジャンルに固執せず、次は実測の誤りに合わせて選ぶ。','']
    lines+=['## 05 実測したオクターブ違いへの学習','','16条件ではギターTP463・FN101・FP122、ベースTP249・FN47・FP72。ギターFP中58音は正解と開始が近く1オクターブ違った。これは原因を特定した件数ではない。正しく対応した704音の終了絶対誤差中央値は約16ms、90%点約53msだった。','', 'そこで同じ推定ギターの近接±12半音候補だけを対象に、真偽の教師あり学習を行った。音高・開始・終了とベースは動かさない。350特徴と近傍差分込み704特徴、各500更新、同じseed23、正例保護重み3、固定閾値候補を比較。選択は局所350特徴・500更新・残すスコア閾値0.35。スコアは校正済み正解確率ではない。正しいオクターブ重奏を一律に消す規則ではない。','']
    table(list(data['octave_regression']['development'].items()))
    lines+=['同じ16開発条件の改善であり、事前に診断を見た集合。14既知バンド条件ではオクターブ補正だけの出力は基準と同じだった。組合せの改善も補完学習による小幅な改善と同じ。原因に合わせた学習に可能性はあるが、一般化の証拠はまだ弱い。追加特徴がある方が必ず良いという結果でもなかった。','']
    table(list(data['octave_regression']['known_regression'].items()))
    lines+=['## 06 音色加工と継続学習の分離','','同じ親・同じ元音源順・切り出し・教師列で600試行。加工なし／EQ＋一部ソロ歪みの比較。','']
    if 'tone' in data:
        table([(k,v['summary']) for k,v in data['tone']['results'].items()])
        lines+=['同じAzure T4・CUDA float16、67開発条件・9データセット群の均等平均。以前のMac float32表と数値を直接混ぜない。元音源・切り出し・正解トークンの600件一致を検査済み。実更新数：'+str(data['tone']['effective_updates'])+'。混合精度計算のスキップにより実更新数は完全一致ではない。','']
        results=data['tone']['results'];c=results['control']['summary'];t=results['tone']['summary'];lines+=['音色増強そのものの対照差は、音符F1 '+f"{t['note_f1']-c['note_f1']:+.4f}"+'、Frame F1 '+f"{t['frame_f1']-c['frame_f1']:+.4f}"+'。加工なし継続学習でも改善しており、親モデルからの差を全部「加工の効果」とは言わない。GOATアンプ音の音符は0.6943→0.7008だが、GuitarSetやFiloBassでは対照より低下した。単一seed、開発検証、実更新数の差もあり、全体への頑健な改善の証拠はまだ弱い。','']
    else:lines+=['比較進行中。実測・回収・停止確認後に更新する。','']
    if 'native_bass_tone' in data:
        lines+=['17既知ベース録音をMac CPU float32で再検査した。両候補は事前に固定し、この結果で再選択・再学習しない。下表は全方式でベース既知の読み取りを揃えた値。GM分類のままの値も公開集計JSONに保存する。','']
        table([(k,v['source_known']) for k,v in data['native_bass_tone'].items()])
        lines+=['加工なし継続学習が加工ありを上回った。ベース既知のFrame・終了込みではMuを上回るが、音符F1はMuの方が高い。しかも既知録音の回帰確認であり、自然なバンドでの全面的な優越の証明ではない。音色加工が常に有効という結論にはしない。','']
    lines+=['## 07 コード名の診断と、不足クラスへの学習','','音符認識とは別のACR評価。GuitarSetのロック・ファンク・ボサノバ各6録音、計18録音の7秒抜粋を使用。両端0.25秒を除く117秒を採点。performedコード（実音符から推定され、根音・区切りは譜面由来）を参照する。指示された簡単なコードと混同しない。同じ進行2の系統・既存の検証集合であり、18個の完全に独立した未知曲ではない。','', '|方式|根音|三和音|構造全体|四和音＋ベース|','|---|---:|---:|---:|---:|']
    chord_rows=[('公開5モデル',data['chord_head']['controls']['published_ensemble']),('公開1モデル',data['chord_head']['controls']['published_single']),('power行追加・学習なし',data['chord_head']['controls']['extended_without_learning'])]
    if data['chord_head']['best']:chord_rows.append(('power行のみ追加学習',data['chord_head']['best']['summary']))
    for label,s in chord_rows:lines.append('|'+label+'|'+'|'.join(f'{s[k]:.4f}' for k in ('root','triads','structure_exact','tetrads_inv'))+'|')
    lines+=['','値は時間一致率。音符F1ではない。根音・三和音・四和音＋ベースはmir_evalのroot/triads/tetrads_inv、構造全体は根音＋全ピッチクラス集合＋ベースの一致。各指標のラベル範囲を同一視しない。','', 'CQT＋CNN/双方向LSTMを固定して、出力層全部（21,616パラメータ）と新しいpower12行だけ（2,316パラメータ）、さらにpower正例重み3を比較。60訓練録音、各500更新、seed31、100/500更新で検証。元の100出力行を固定する条件は元の重みを保つが、候補の競合により予測の不変までは保証しない。','', '選択はpower行のみ・重み3・500更新。公開5モデル比で構造全体0.2216→0.2354だが、公開1モデルの0.2275からの学習効果は小さい。四和音＋ベースは5モデル0.2550に対し0.2535とわずかに低下した。自動で標準モデルへ置き換えず、比較用候補として出力する。','', '出力層全部を学習する条件は根音も悪化した。最初の2条件は音声の再サンプル処理が5モデル対照と違ったため、同じ前処理による主比較から除外・記録を保存。正式比較はFFmpegの22,050HzモノラルPCM16→CQTへ統一した3条件である。','', '語彙だけでも限界がある。同じ117秒で表現可能な参照構造は標準語彙67.2%、powerを含む語彙82.1%。さらに学習後も、power根音＋三和音の正しい時間は参照17.50秒の約0.72秒（4.1%）に過ぎない。C5を出力できるようにしたことと、正確に認識できることは別。根音を当てるより、構成音の有無・省略・ベース・テンションの区別が難しい。','', '次は実音符ラベルから曖昧でないpower区間を選ぶ教材、第三音あり／なしの難しい対照、表現できない省略コード、特徴抽出器を含む追加学習を比較する。音符から作る仮の和音ラベルは、人が付けた和声ラベルと区別する。BTCへの置換だけで語彙の不足が解決すると考えない。','']
    lines+=['## 08 LoRA全実行の一覧','','選択stepは各実行の開発検証による。異なる実行で検証集合や初期値が違うため、この一覧だけでは公平な順位にならない。','', '|実行名|選択step|更新範囲|追加パラメータ|','|---|---:|---|---:|']
    for r in runs:lines.append(f"|{r['name']}|{r['selected_step']}|{r['scope']} / {r['targets']}|{r['trainable_parameters']}|")
    cleanup=data['azure_cleanup']
    if cleanup.get('deallocated'):
        when=datetime.fromisoformat(cleanup['observed_at']).strftime('%m月%d日 %H:%M（日本時間）');ending=f"Azureは{when}に停止済み（割り当て解除）を再確認。{cleanup['verified_checkpoints']:,}チェックポイントと、監査対象{cleanup['verified_files']:,}ファイル（約{cleanup['verified_bytes']/1e9:.2f}GB）の回収・照合を確認した。Macだけで作成した追加実験は別のローカル保存対象。"
    else:ending='Azure比較・回収・割り当て解除は進行中。最終版ではない。'
    lines+=['','## 09 再現と保管','','各表の原本JSONとSHA256は docs/measurements/review_20261119.json に記録。正解音符・音源・研究用重み・認証はGitへ公開しない。', '',ending,'', '詳細な目的・失敗・歴史は RESEARCH_REVIEW_20261119.md を参照。']
    Path('docs/LATEST_RESULTS_20261119.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('LATEST_REVIEW_READY',len(runs),'LoRA runs')


if __name__=='__main__':main()
