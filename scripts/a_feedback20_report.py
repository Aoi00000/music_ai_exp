"""Aggregate Nov20 experiments without publishing audio, notes, or model weights."""
from pathlib import Path
import json,hashlib,argparse
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.load(open(p))
def main(make_pdf=False):
    paths=['reports/residual_v1/cache_counts.json','reports/residual_v1/sampling_audit.json','reports/residual_identity_v1/selection.json',
      'reports/residual_tree_v2/final_selection.json','reports/residual_fixed_review_v1/comparison.json',
      'reports/mu_adapt_v1/selection.json','reports/mu_adapt_v1/review.json','reports/mu_native_v2/selection.json','reports/mu_native_v2/review.json',
      'reports/mu_consensus_final_v4/selection.json','reports/mu_consensus_final_v4/review.json','reports/azure_session_cleanup.json','reports/azure_deallocated_20261120.json','reports/residual_calibration_v3/selection.json','reports/provenance_review_20261120.json','reports/mu_canonical_v3/selection.json','reports/mu_canonical_v3/review.json','reports/octave_missing_diagnostic_20261120.json','reports/artifact_audit_20261120.json']
    mu={};integrity={}
    for family in ('mu_adapt_v1','mu_native_v2','mu_canonical_v3'):
        s=read('reports/'+family+'/selection.json');r=read('reports/'+family+'/review.json')
        mu[family]=dict(selected_on_internal=s['chosen'],internal_baseline=s['baseline'],internal_choices=s['choices'],reused16={k:v['summary'] for k,v in r['results'].items()})
        for arm in ('gold','gold_tone'):integrity[family+'/'+arm]=read('runs/'+family+'/'+arm+'/completed.json')
    rr=read(paths[4]);nn=read(paths[2]);trees=read(paths[3]);counts=read(paths[0])
    if any(v['selected_on_internal']!='baseline' for v in mu.values()) or read(paths[9])['best'] is not None:raise ValueError('New adopted result: revise narrative before publication')
    d=dict(date='2026-11-20',nn_runs=12,nn_updates=9600,tree_fits=6,tree_iterations_per_fit=200,mu_lora_runs=6,mu_lora_updates=720,mu_gradient_accumulation=2,
       no_new_reinforcement_learning=True,no_new_external_dataset_download=True,training_target_counts=counts,sampling=read(paths[1]),
       nn_calibration_baseline=nn['baseline'],tree_calibration_baseline=trees['baseline'],low_score_calibration=dict(identity_selected=read(paths[13])['identity_selected'],winner=read(paths[13])['winner'],settings_compared=486),nn_final_policies=[dict(arm=a['arm'],thresholds=a['best']['thresholds'],strict_gain=a['calibration_strict_gain']) for a in nn['arms']],
       tree_final_policies=[dict(arm=a['arm'],thresholds=a['final_thresholds'],strict_gain=a['strict_gain']) for a in trees['arms']],
       tree_seeds_caveat=trees['seeds_note'],candidate_diagnostic=dict(summary={k:{a:v[a] for a in ('baseline','unfiltered_union','candidate_oracle_diagnostic','chosen_plus_c7')} for k,v in rr['summary'].items()},coverage=rr['coverage']),
       mu=mu,training_integrity=integrity,provenance_correction=read(paths[14]),octave_error_diagnostic=read(paths[17]),artifact_audit={k:read(paths[18])[k] for k in ('files','bytes','all_sha256_matched')},consensus=dict(best=read(paths[9])['best'],baseline=read(paths[10])['baseline'],selected=read(paths[10])['selected']['summary']),
       dataset_roles='All added head/adapter training rows from acquired training-only sources. GuitarSet8+TECHS32 added for heads; source UUID/GOAT-calibration protection retained. Eleven final Slakh sources remain unscored.',
       automatic_application_approved=False,app_default_changed=False,test_used=False,azure_cleanup=read(paths[11]),azure_status=read(paths[12]),sources={p:sha(p) for p in paths})
    save('docs/measurements/feedback_20261120.json',d)
    columns=['note_f1','frame_f1','active_exact_match','offset_f1']
    def row(label,m):return '| '+label+' | '+' | '.join(f'{m[k]*100:.3f}%' for k in columns)+' |'
    table=['| 同じ計算条件の16条件 | 音符F1 | Frame F1 | 全音一致 | 終了F1 |','| --- | --- | --- | --- | --- |']
    table.append(row('学習なし',mu['mu_canonical_v3']['reused16']['baseline']))
    for family,prefix in [('mu_canonical_v3','一律細分類・再対照'),('mu_native_v2','元Muの細分類')]:
        for arm,suffix in [('gold','加工なし'),('gold_tone','音色加工')]:table.append(row(prefix+'・'+suffix,mu[family]['reused16'][arm]))
    appendix='''
<!-- page -->
## 7. Mu追加学習の結果：改善する指標だけで採用しない

'''+ '\n'.join(table)+'''

Mu本体のLoRAは初回予備2・細分類保持2・条件を揃えた再対照2の計6回、各120更新、計720更新。表は数値計算条件を揃えた再対照と細分類保持の主比較である。全6回で元の大きな重みが変わらないこと、8層すべてで追加行列が実際に更新されたこと、教材の抽選が同じことを確認した。

元重みを固定しても、LoRAの追加出力で予測は変わるため、精度が下がる可能性はある。Frame F1はわずかに上がったが、音符F1・全音一致・終了F1が下がった。細分類を元Muに合わせても、改善は確認できなかった。最初の教師の細分類だけが悪化の原因だったとは結論しない。音色加工の追加効果も今回の範囲では確認できない。

内部5条件の数値も、実際に実行した設定・重みと対応させて記録した。初回予備の一律細分類は音符F1 50.418%、元Muの細分類は49.731%で、いずれも基準52.737%を下回った。初回予備は勾配スケールの初期値が違うため、細分類だけの効果を分離する主比較に混ぜない。学習後の終了だけを使う64設定の比較でも、内部で4指標を保護しながら厳密に改善する設定はなかった。これも新しい64回の学習とは数えない。

この表の基準は、CUDAでfloat32重み・float16計算を使って新しく推論した値。以前の保存済み基準とはごく小さな終了時刻差があり、Frameや全音一致が少し違う。各LoRAは同じ条件で再推論した基準と比べ、違う計算条件を一本の精度向上曲線に混ぜない。

今回の追加は、NN 12学習、木6fit、Mu LoRA 6学習の計24回。木の反復と勾配更新は異なるので合算しない。新しい比較案が従来案を上回ったという結論には至らず、既存の基準・前回のC5/C7比較案を維持する。前回の限定的な改善を今回の成功として数え直さない。

<!-- page -->
## 8. 発表への価値、次の優先順位、Macへの影響

研究上の価値は、候補数の不足と選別の難しさを分け、教師ありの補助分類からMu本体のLoRAまで、理由を持って比較した点にある。候補を増やしても真の補完は少なく、音色を増やしても新しい奏者は増えない。細分類の仮説を立て、他の記号と音声を固定して確認し、悪化した方式を採用しないという過程を発表できる。

次の優先1：余分な高い音と本物のオクターブ重音を区別する教材に絞る。取得済みの単音素材と音色加工から、単音の倍音／実際の2音を比較する合成教材を検討する。弦別音声は取得状況を確認して補充する。57/58の診断から立てた次の仮説で、今回まだ学習した成果ではない。見落としは候補生成・分離欠落・同音重複・短音の別に分析する。正解ステムを使う診断は到達可能性を見るために限定し、アプリで使える結果と区別する。

優先2：Muの更新範囲が最後の8層のAttention出力だけだったこと、1 seed・120更新の小さな試行だったことを踏まえる。次の追加学習では、同じ音符教師を保ち、更新範囲・学習量・元モデル保持の強さのうち一つずつ変える。音色増強だけを無目的に増やすより、独立した奏者や短音・ミュートの正例を補う。更新を広げると良くなるとは未検証である。

優先3：11/21までに追加比較を絞り、11/22に方式・重み・設定を固定して保留11元曲で初回比較する。そこで悪化したら基準を採用し、失敗例として説明する。11/23にBとデモを固定し、11/24に練習、11/25に発表する。コード名・拍・運指・TABの精度は別に検証し、今回の音符指標で証明しない。

Bのアプリ・元の12秒ZIP・基準JSON契約・既定の認識モデルは変更していない。今回の不採用モデル一式をBへ移す必要はない。Bは従来の基準と任意補正案の読込・再生同期・1か所編集・再保存・MusicXML出力の結合確認を継続できる。

全学習の成果をMacへ取り出し、チェックポイントのSHA256を照合した。Azureは最後に「停止済み（割り当て解除）」を直接確認済み。独立した終了処理で、学習・比較後に回収・検証・割り当て解除を行った。仮想マシンを稼働したまま次回を待ってはいない。

実測の集計は docs/measurements/feedback_20261120.json。再現コードは scripts/a_residual_cycle.py、a_residual_tree.py、a_mu_adapt.py、a_mu_native_cycle.py、a_mu_consensus.py。学習済み重み・音源・正解音符・認証情報は公開Gitには含めない。
'''
    p=Path('docs/FEEDBACK_CYCLE_20261120.md');source=p.read_text().split('<!-- GENERATED_RESULTS -->')[0];p.write_text(source+'<!-- GENERATED_RESULTS -->\n'+appendix)
    if make_pdf:
        from scripts.a_keep_report import pdf
        pdf('docs/FEEDBACK_CYCLE_20261120.md','output/pdf/band2tab_feedback_20261120.pdf','2026-11-20',three_column_widths=(.38,.34,.28))
    print('FEEDBACK20_PUBLISHED',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pdf',action='store_true');a=p.parse_args();main(a.pdf)
