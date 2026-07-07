"""
Seq-length sweep experiment.
Trains the Multi-Tower BiGRU at seq_length = 5, 9, 13
and prints a comparison table.
"""
import os, sys, json, joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

np.random.seed(42); torch.manual_seed(42)

# ── Feature groups (same as multi_tower_model.py) ──────────────────────────
TOWER_RVI = ['raw_RVI','RVI_lag_6','RVI_lag_12','RVI_lead_6','RVI_lead_12','RVI_velocity_6','RVI_velocity_12']
TOWER_VV  = ['raw_VV', 'VV_lag_6', 'VV_lag_12', 'VV_lead_6', 'VV_lead_12', 'VV_velocity_6', 'VV_velocity_12']
TOWER_VH  = ['raw_VH', 'VH_lag_6', 'VH_lag_12', 'VH_lead_6', 'VH_lead_12', 'VH_velocity_6', 'VH_velocity_12']
TOWER_WX  = ['Rainfall_15d_sum','MaxTemp_7d_avg','MinTemp_7d_avg']
TOWER_CTX = ['doy_sin','doy_cos','is_kharif','is_rabi','is_zaid','expected_baseline_ndvi']
ALL_COLS  = TOWER_RVI + TOWER_VV + TOWER_VH + TOWER_WX + TOWER_CTX
TARGET    = 'raw_NDVI'

def tower_idx():
    a=len(TOWER_RVI); b=a+len(TOWER_VV); c=b+len(TOWER_VH); d=c+len(TOWER_WX)
    return list(range(0,a)), list(range(a,b)), list(range(b,c)), list(range(c,d)), list(range(d,len(ALL_COLS)))

class DS(Dataset):
    def __init__(self, X, y):
        ri,vi,hi,wi,ci = tower_idx()
        self.xr=torch.tensor(X[:,:,ri],dtype=torch.float32)
        self.xv=torch.tensor(X[:,:,vi],dtype=torch.float32)
        self.xh=torch.tensor(X[:,:,hi],dtype=torch.float32)
        self.xw=torch.tensor(X[:,:,wi],dtype=torch.float32)
        self.xc=torch.tensor(X[:,:,ci],dtype=torch.float32)
        self.y =torch.tensor(y,dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.xr[i],self.xv[i],self.xh[i],self.xw[i],self.xc[i],self.y[i]

class MT(nn.Module):
    def __init__(self,seq):
        super().__init__()
        self.seq=seq
        self.tr=nn.GRU(7,48,1,batch_first=True,bidirectional=True)
        self.tv=nn.GRU(7,32,1,batch_first=True,bidirectional=True)
        self.th=nn.GRU(7,32,1,batch_first=True,bidirectional=True)
        self.tw=nn.GRU(3,24,1,batch_first=True,bidirectional=True)
        self.tc=nn.GRU(6,24,1,batch_first=True,bidirectional=True)
        self.norm=nn.LayerNorm(320)
        self.fc1=nn.Linear(320,128); self.fc2=nn.Linear(128,32); self.fo=nn.Linear(32,1)
        self.relu=nn.ReLU(); self.d1=nn.Dropout(0.3); self.d2=nn.Dropout(0.2)
    def cx(self,g): return g[:,self.seq//2,:]
    def forward(self,r,v,h,w,c):
        h_=torch.cat([self.cx(self.tr(r)[0]),self.cx(self.tv(v)[0]),
                       self.cx(self.th(h)[0]),self.cx(self.tw(w)[0]),self.cx(self.tc(c)[0])],dim=-1)
        h_=self.norm(h_)
        return self.fo(self.d2(self.relu(self.fc2(self.d1(self.relu(self.fc1(h_)))))))

def make_seqs(df, seq):
    X,y=[],[]
    off=seq//2
    for _,g in df.groupby('task_id'):
        g=g.sort_values('date').reset_index(drop=True)
        F=g[ALL_COLS].values; T=g[TARGET].values; n=len(g)
        for i in range(n):
            s=[]
            for idx in range(i-off, i+(seq-1-off)+1):
                s.append(F[max(0,min(idx,n-1))])
            X.append(np.array(s)); y.append(T[i])
    return np.array(X),np.array(y)

def run(seq_len, df_tr, df_te, device):
    print(f"\n{'='*50}")
    print(f"  seq_length = {seq_len}  (±{seq_len//2} satellite visits)")
    print(f"  Calendar context ≈ {seq_len//2 * 6}–{seq_len//2 * 12} days")
    print(f"{'='*50}")

    Xtr,ytr = make_seqs(df_tr, seq_len)
    Xte,yte = make_seqs(df_te, seq_len)
    print(f"  Sequences — train: {Xtr.shape}  test: {Xte.shape}")

    trl = DataLoader(DS(Xtr,ytr), batch_size=64,  shuffle=True)
    tel = DataLoader(DS(Xte,yte), batch_size=256, shuffle=False)

    m = MT(seq_len).to(device)
    opt = optim.AdamW(m.parameters(), lr=0.0005, weight_decay=1e-3)
    crit = nn.MSELoss()
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40, eta_min=1e-6)
    best, bst, pat = 1e9, None, 0

    for ep in range(40):
        m.train()
        for r,v,h,w,c,by in trl:
            r,v,h,w,c,by = r.to(device),v.to(device),h.to(device),w.to(device),c.to(device),by.to(device)
            opt.zero_grad(); loss=crit(m(r,v,h,w,c),by); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        sch.step()
        m.eval(); vl=0.0
        with torch.no_grad():
            for r,v,h,w,c,by in tel:
                r,v,h,w,c,by = r.to(device),v.to(device),h.to(device),w.to(device),c.to(device),by.to(device)
                vl += crit(m(r,v,h,w,c),by).item()*by.size(0)
        vl /= len(tel.dataset)
        if vl < best: best=vl; bst={k:v_.clone() for k,v_ in m.state_dict().items()}; pat=0
        else: pat+=1
        tag = " ← best" if pat==0 else f" (pat {pat}/8)"
        print(f"  Ep {ep+1:02d}/40 | val MSE {vl:.5f}{tag}")
        if pat>=8: print(f"  Early stop at epoch {ep+1}"); break

    m.load_state_dict(bst); m.eval()
    preds,trues=[],[]
    with torch.no_grad():
        for r,v,h,w,c,by in tel:
            r,v,h,w,c = r.to(device),v.to(device),h.to(device),w.to(device),c.to(device)
            preds.append(m(r,v,h,w,c).cpu().numpy()); trues.append(by.numpy())
    yp = np.clip(np.vstack(preds).squeeze(),0,1)
    yt = np.vstack(trues).squeeze()
    r2=r2_score(yt,yp); rmse=mean_squared_error(yt,yp)**0.5; mae=mean_absolute_error(yt,yp)
    print(f"\n  ✅ R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")
    return {"seq_length":seq_len,"r2":r2,"rmse":rmse,"mae":mae}

# ── Main ───────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(script_dir,"../data/model_training_dataset.csv"))

df['RVI_velocity_6'] =df['raw_RVI']-df['RVI_lag_6'];  df['RVI_velocity_12']=df['raw_RVI']-df['RVI_lag_12']
df['VV_velocity_6']  =df['raw_VV'] -df['VV_lag_6'];   df['VV_velocity_12'] =df['raw_VV'] -df['VV_lag_12']
df['VH_velocity_6']  =df['raw_VH'] -df['VH_lag_6'];   df['VH_velocity_12'] =df['raw_VH'] -df['VH_lag_12']
t0=df[['water','bare','snow_and_ice']].sum(1)
t1=df[['shrub_and_scrub','built','flooded_vegetation','crops']].sum(1)
t2=df[['grass','trees']].sum(1)
df['expected_baseline_ndvi']=0.0531*t0+0.3817*t1+0.6207*t2
df=df.dropna(subset=[TARGET])
df=df.groupby('task_id',group_keys=False).apply(lambda x:x.sort_values('date')).reset_index(drop=True)

gss=GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
tri,tei=next(gss.split(df,df[TARGET],groups=df['task_id']))
df_tr=df.iloc[tri].copy(); df_te=df.iloc[tei].copy()

sc=StandardScaler()
df_tr[ALL_COLS]=sc.fit_transform(df_tr[ALL_COLS]); df_te[ALL_COLS]=sc.transform(df_te[ALL_COLS])

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device} | Train parcels: {df_tr['task_id'].nunique()} | Test: {df_te['task_id'].nunique()}")
print(f"\nBaseline (single BiGRU, seq=5): R²=0.8381  RMSE=0.1105  MAE=0.0703")

results=[]
for sl in [5, 9, 13]:
    results.append(run(sl, df_tr, df_te, device))

print("\n\n" + "="*60)
print("SEQ-LENGTH COMPARISON SUMMARY")
print("="*60)
print(f"{'seq_len':>8} | {'context':>18} | {'R²':>7} | {'RMSE':>7} | {'MAE':>7}")
print("-"*60)
print(f"{'5 (base)':>8} | {'~30-60 days':>18} | {'0.8381':>7} | {'0.1105':>7} | {'0.0703':>7}  ← single BiGRU baseline")
for r in results:
    ctx=f"~{r['seq_length']//2*6}–{r['seq_length']//2*12} days"
    print(f"{r['seq_length']:>8} | {ctx:>18} | {r['r2']:>7.4f} | {r['rmse']:>7.4f} | {r['mae']:>7.4f}")
print("="*60)
