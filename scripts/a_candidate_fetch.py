"""New source UUIDs for candidate learning; no prior test reused as a fresh test."""
from scripts.a_temporal_fetch import main

if __name__=='__main__':
    main(root='data/raw/slakh_candidate_v2',report='reports/candidate_v2',
         caps={'train':32,'validation':8,'test':12},
         exclude_acquisitions=['reports/temporal_v1/source_acquisition.json'],
         excluded_plugins=['scarbee_jay_bass_slap_both.nkm'],
         sort_salt='candidate-v2-20261118-',budget_gib=4)
