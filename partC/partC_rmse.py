#!/usr/bin/env python3
# Compute positional RMSE between estimated camera positions and ground truth.

import argparse, csv, math
import numpy as np

def load_est(p):
    ts,P=[],[]
    with open(p,newline="") as f:
        r=csv.DictReader(f)
        for row in r:
            ts.append(float(row["timestamp"]))
            P.append([float(row["x"]),float(row["y"]),float(row["z"])])
    return np.array(ts), np.array(P)

def load_gt(p):
    ts,P=[],[]
    with open(p,newline="") as f:
        r=csv.DictReader(f); cols=r.fieldnames
        tkey="timestamp" if "timestamp" in cols else ("frame" if "frame" in cols else None)
        for row in r:
            ts.append(float(row[tkey]) if tkey else len(ts))
            P.append([float(row["x"]),float(row["y"]),float(row["z"])])
    return np.array(ts), np.array(P)

def align(ts_e,P_e,ts_g,P_g):
    if len(ts_g)==1: return np.arange(len(ts_e)), np.zeros(len(ts_e),int)
    idx=[int(np.argmin(np.abs(ts_g-t))) for t in ts_e]
    return np.arange(len(ts_e)), np.array(idx)

def main():
    ap=argparse.ArgumentParser(description="Positional RMSE (meters)")
    ap.add_argument("--est", required=True)
    ap.add_argument("--gt", required=True)
    args=ap.parse_args()
    ts_e,P_e=load_est(args.est); ts_g,P_g=load_gt(args.gt)
    I,J=align(ts_e,P_e,ts_g,P_g); dif=P_e[I]-P_g[J]
    rmse=float(np.sqrt(np.mean(np.sum(dif**2,axis=1))))
    print(f"Positional RMSE (m): {rmse:.6f} over {len(I)} samples")

if __name__=="__main__": main()
