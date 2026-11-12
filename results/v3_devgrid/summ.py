import json, sys, pandas as pd
df = pd.DataFrame(sum([json.load(open(f)) for f in sys.argv[1:]], []))
piv = df.groupby(["ds","cfg"]).ap.mean().unstack(0)
t = df.groupby("cfg").t.mean()
rank = piv.rank(ascending=False, axis=0).mean(axis=1)
piv["avgAP"] = piv.mean(axis=1); piv["avgRank"] = rank; piv["fit_s"] = t
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
piv.columns = [c[:9] for c in piv.columns]
print(piv.sort_values("avgRank").round(3).to_string())
