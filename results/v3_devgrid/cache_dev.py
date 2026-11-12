import sys, numpy as np, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/Volumes/WORK/__Articles/Cross-Balance-v2")
from benchmarks.datasets import load_synthetic
from benchmarks.real_datasets import REAL_DATASETS, stratified_subsample
import os; os.makedirs("cache", exist_ok=True)
for k in ["99-1","98-2","95-5"]:
    X,y = load_synthetic(k); np.savez(f"cache/syn_{k}.npz", X=X, y=y)
for k in ["credit_card_fraud","synthetic_financial_fraud","ailerons","communities_crime","bank_marketing","default_credit_card_clients","mammographic","segment0","shuttle_2_vs_5","fraud_detection_transactions","mushrooms","credit_card_fraud_2023"]:
    X,y = REAL_DATASETS[k](); X,y = stratified_subsample(X,y,50000,random_state=42)
    np.savez(f"cache/{k}.npz", X=X.astype(np.float32), y=y); print(k, X.shape, y.mean())
