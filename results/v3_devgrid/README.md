# v3 development grid (design freeze evidence)

Three grids on the 10 development datasets only (3 synthetic + credit_card_fraud,
PaySim, ailerons, communities_crime, bank_marketing, default_credit_card_clients,
mammographic), stratified 3-fold CV, seed 0, AP as the criterion. `proto.py` is
the prototype used (`CBProto`); `summary.txt` is the ranked table
(`python summ.py dev1.json dev2.json dev3.json`). Chosen configuration:
`sp_mix_r4` = self-paced sampling, alternating HistGB/ExtraTrees members,
ratio 4, K = 50, cap 3000, uniform weights -> `CrossBalanceClassifier()` v0.2 defaults.
Scripts expect a `cache/` folder created by `cache_dev.py`.
