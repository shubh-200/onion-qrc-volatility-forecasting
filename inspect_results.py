import json, sys
sys.path.insert(0,'.')
from pathlib import Path
# Check simulator results
for jf in sorted(Path('prototype/results/phase3').rglob('*.json')):
    d = json.load(open(jf))
    for r in d.get('results', []):
        preds = r.get('test_predictions')
        has_p = 'yes' if preds else 'no'
        n_p = len(preds) if preds else 0
        model = r.get('model','?')
        nq = r.get('n_qubits','—')
        topo = r.get('topology','—')
        seed = r.get('seed','—')
        print(f"  {model} N={nq} {topo} seed={seed}: has_predictions={has_p}, n_preds={n_p}")
