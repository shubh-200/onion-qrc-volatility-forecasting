import json

d = json.load(open('artifacts/manifests/statistical_analysis.json'))

print("=== 1. Diebold-Mariano Tests vs HAR-Ridge (QLIKE Loss) ===")
print(f"{'Model':35s} | {'DM Stat':>10s} | {'p-value':>10s} | {'Interpretation':>25s}")
print("-" * 88)
for k, v in d['diebold_mariano_vs_HAR_Ridge'].items():
    stat = v['dm_stat']
    pval = v['p_value']
    if stat > 0 and pval < 0.10:
        interp = "HAR-Ridge significantly better"
    elif stat < 0 and pval < 0.10:
        interp = "Model significantly better"
    else:
        interp = "No stat. diff vs HAR"
    print(f"{k:35s} | {stat:10.4f} | {pval:10.4f} | {interp:>25s}")

print("\n=== 2. Diebold-Mariano Tests vs ESN-500 (Classical Reservoir Benchmark) ===")
print(f"{'Model':35s} | {'DM Stat':>10s} | {'p-value':>10s} | {'Interpretation':>25s}")
print("-" * 88)
for k, v in d['diebold_mariano_vs_ESN_500'].items():
    stat = v['dm_stat']
    pval = v['p_value']
    if stat < 0 and pval < 0.10:
        interp = "Model significantly beats ESN"
    elif stat > 0 and pval < 0.10:
        interp = "ESN-500 significantly better"
    else:
        interp = "No stat. diff vs ESN"
    print(f"{k:35s} | {stat:10.4f} | {pval:10.4f} | {interp:>25s}")
