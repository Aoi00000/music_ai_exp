import copy,unittest
from scripts.a_mu_consensus import transfer

class LearnedReleaseTransfer(unittest.TestCase):
    def test_onsets_pitches_counts_and_original_stay_protected(self):
        n=lambda p,a,b:dict(pitch=p,onset=a,offset=b,velocity=100)
        base={'guitar':[n(60,.5,1.),n(64,1.2,1.6)],'bass':[n(40,.5,1.5)]};before=copy.deepcopy(base)
        learned={'guitar':[n(60,.51,1.1),n(64,1.25,1.7),n(67,1.,2.)],'bass':[n(40,.5,2.)]}
        out=transfer(base,learned,(.5,1.))
        self.assertEqual(base,before);self.assertAlmostEqual(out['guitar'][0]['offset'],1.05)
        self.assertEqual(out['guitar'][1],base['guitar'][1]);self.assertEqual(out['bass'],base['bass'])
        for p in base:
            self.assertEqual([(x['pitch'],x['onset'],x['velocity']) for x in out[p]],[(x['pitch'],x['onset'],x['velocity']) for x in base[p]])
if __name__=='__main__':unittest.main()
