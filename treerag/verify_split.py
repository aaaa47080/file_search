import os

files = [
    'tree_models.py',
    '_phase1_structure.py',
    '_phase2_verify.py',
    '_phase3_tree.py',
    '_phase4_summary.py',
    'tree_index.py'
]

total = 0
for f in files:
    if os.path.exists(f):
        with open(f) as fp:
            lines = len(fp.readlines())
        total += lines
        print(f"{f:30s} {lines:5d} lines")
    else:
        print(f"{f:30s} NOT FOUND")

print("-" * 45)
print(f"{'TOTAL':30s} {total:5d} lines")
print("\nTarget: 1480 lines → ~1500-1520 lines (slight growth due to imports)")
