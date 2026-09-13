"""Frozen FINAL.md analysis. Refuses partial runs and unverified transport bytes."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import core as C
from worker import atomic, fingerprint
from run import units, valid


def variance(x):
    return float(np.var(x, axis=0, ddof=1).sum())


def ratio(n, d):
    return None if d <= 0 else float(n / d)


def metrics(x, y, z, phi0, phi1):
    """Input is R by p; independent uses z-x, coupled uses y-x."""
    truth = phi1 - phi0
    a, b, c = x - phi0, y - phi1, z - phi1
    denom = float(np.sum(phi0**2) + np.sum(phi1**2) + 1e-12)
    output = {}
    for arm, estimate in [('crn', y-x), ('independent', z-x)]:
        error = estimate - truth
        mse = float(np.mean(np.sum(error**2, axis=1)))
        bias2 = float(np.sum(error.mean(axis=0)**2))
        var = variance(error)
        identity = mse - bias2 - (len(error)-1)/len(error)*var
        assert abs(identity) <= 1e-10 * max(1., mse), 'MSE decomposition'
        top = np.argsort(-np.abs(truth), kind='stable')[:5]
        selected = np.argsort(-np.abs(estimate), axis=1, kind='stable')[:, :5]
        mismatch = float(np.mean([1-len(set(row)&set(top))/5 for row in selected]))
        output.update({f'mse_{arm}': mse, f'nmse_{arm}': mse/denom,
                       f'bias2_plugin_{arm}': bias2, f'bias2_corrected_{arm}': bias2-var/len(error),
                       f'variance_{arm}': var, f'top5_mismatch_{arm}': mismatch})
    marginal = float(np.mean(np.sum(b*b-c*c, axis=1)))
    cross = float(-2*np.mean(np.sum(a*(b-c), axis=1)))
    residual = output['mse_crn']-output['mse_independent']-marginal-cross
    assert abs(residual) < 1e-10, 'finite coupling identity'
    output.update(normalizer=denom, mse_difference_marginal=marginal,
                  mse_difference_cross=cross, mse_identity_residual=residual)
    return output


def bootstrap_cluster_ratio(clusters, repeats=5000, seed=684217):
    groups = [g[['nmse_crn','nmse_independent']].to_numpy()
              for _, g in clusters.groupby('dataset', sort=True)]
    rng = np.random.default_rng(seed)
    draws = np.zeros((repeats, 2))
    for group in groups:
        ix = rng.integers(0, len(group), size=(repeats, len(group)))
        draws += group[ix].sum(axis=1)
    if np.any(draws[:, 1] <= 0):
        return {'ci95': None, 'undefined_draws': int(np.sum(draws[:,1]<=0))}
    values = draws[:,0]/draws[:,1]
    return {'ci95': np.percentile(values, [2.5,97.5]).tolist(), 'undefined_draws': 0}


def primary(e1, config):
    output = {}
    for method in ['ks_complement','leverage']:
        cells = e1[(e1.method==method)&(e1.scenario!='null')]
        strata = cells.groupby(['dataset','split','model'], sort=True)[['nmse_crn','nmse_independent']].mean()
        assert len(strata)==len(config['datasets'])*len(config['splits'])*len(config['models'])
        clusters = strata.groupby(['dataset','split']).mean().reset_index()
        assert all(n==len(config['splits']) for n in clusters.groupby('dataset').size())
        pooled = ratio(clusters.nmse_crn.sum(), clusters.nmse_independent.sum())
        adult = clusters[clusters.dataset=='adult']
        adult_point = ratio(adult.nmse_crn.sum(), adult.nmse_independent.sum())
        ci = bootstrap_cluster_ratio(clusters, config['bootstrap_replicates'])
        adult_ci = bootstrap_cluster_ratio(adult, config['bootstrap_replicates'])
        points = {f'{d}/{m}':ratio(g.nmse_crn.mean(),g.nmse_independent.mean())
                  for (d,m),g in strata.reset_index().groupby(['dataset','model'],sort=True)}
        checks = {'pooled_upper_below_one': ci['ci95'] is not None and ci['ci95'][1]<1,
                  'three_of_four_strata': sum(v is not None and v<1 for v in points.values())>=3,
                  'adult_upper_below_one': adult_ci['ci95'] is not None and adult_ci['ci95'][1]<1,
                  'both_adult_models': all(points[f'adult/{m}'] is not None and points[f'adult/{m}']<1 for m in config['models'])}
        output[method] = {'pooled_ratio':pooled, 'pooled_bootstrap':ci,
                          'adult_ratio':adult_point, 'adult_bootstrap':adult_ci,
                          'strata':points, 'checks':checks, 'supported':all(checks.values())}
    return {'status': 'SUPPORTED_BOUNDED' if all(x['supported'] for x in output.values()) else 'NOT_SUPPORTED',
            'methods':output, 'interval_scope':'dataset-stratified repeated-split resampling; not independent populations',
            'seed':684217, 'replicates':config['bootstrap_replicates'],
            'historical_gate_c':'KILL_PIVOT', 'detector_superiority':False}


def analyze(work, config_path, destination):
    config=json.loads(config_path.read_text())
    status=json.loads((work/'run_status.json').read_text())
    fp=status['fingerprint']
    approved=json.loads((C.ROOT/'provenance.json').read_text())
    assert C.digest(config_path)==approved['config_sha256'], 'public configuration changed'
    assert fp==approved['original_scientific_fingerprint'], 'saved run identity changed'
    planned=units(config)
    assert status['status']=='COMPLETED' and status['exit_code']==0
    assert status['completed']==status['total']==len(planned)==720, 'complete primary coverage required'
    assert status['fingerprint']==fp
    manifest=json.loads((work/'SNAPSHOT_MANIFEST.json').read_text())
    assert manifest['source_exit_code']==0 and manifest['scientific_fingerprint']==fp
    for name,h in manifest['files'].items():
        path=(work/name).resolve()
        assert path.is_relative_to(work.resolve()) and C.digest(path)==h, 'transport hash '+name
    e1=[]; e2=[]; costs=[]; windows=[]
    env=json.loads((work/'environment.json').read_text())
    for unit in planned:
        attempt=valid(work,unit,fp)
        assert attempt is not None, 'missing or invalid unit '+unit['id']
        completed=json.loads((attempt/'completed.json').read_text())
        assert completed['environment']==env, 'mixed environments'
        doc=json.loads((attempt/'records.json').read_text()); recs=doc['records']
        assert len(recs)==768 and doc['unit']==unit
        p={'adult':14,'electricity':8}[unit['dataset']]
        methods=config['methods'] if unit['arm']=='E1' else config['window_methods']
        budgets=config['budgets'][unit['dataset']] if unit['arm']=='E1' else config['budgets'][unit['dataset']][:1]
        draws=config['draws'] if unit['arm']=='E1' else config['window_draws']
        meta={k:v for k,v in unit.items() if k!='id'}; meta['unit_id']=unit['id']
        with np.load(attempt/'raw.npz') as data:
            assert all(np.isfinite(data[k]).all() for k in data.files)
            phi0,phi1=data['oracle_phi0'],data['oracle_phi1']
            expected_phi=(p,) if unit['arm']=='E1' else (config['window_size'],p)
            assert phi0.shape==phi1.shape==expected_phi
            for method in methods:
                for budget in budgets:
                    raw=data[f'{method}__{budget}']
                    expected=(draws,3,p) if unit['arm']=='E1' else (draws,config['window_size'],3,p)
                    assert raw.shape==expected
                    selected=[r for r in recs if r['method']==method and r['budget']==budget]
                    expected_keys={(d,i,j) for d in range(draws)
                                   for i in range(1 if unit['arm']=='E1' else config['window_size']) for j in range(3)}
                    observed={(r['draw'],r.get('position',0),r['role']) for r in selected}
                    assert len(selected)==len(expected_keys) and observed==expected_keys
                    rank_target=p-1 if method!='poly3' else p+math.comb(p,3)-1
                    for r in selected:
                        assert r['rows']==budget and 2<=r['unique_rows']<=budget
                        assert r['ranks'] and all(0<=x['rank']<=x['columns'] for x in r['ranks'])
                        costs.append({**meta,'method':method,'budget':budget,'draw':r['draw'],
                                      'position':r.get('position',0),'role':r['role'],
                                      'rows':r['rows'],'calls':r['calls'],'unique_rows':r['unique_rows'],
                                      'rank':min(x['rank'] for x in r['ranks']),'rank_target':rank_target,
                                      'additional_rank_loss':any(x['rank']<rank_target for x in r['ranks'])})
                    for d,i,_ in expected_keys:
                        rr={r['role']:r for r in selected if r['draw']==d and r.get('position',0)==i}
                        assert rr[0]['seed']==rr[1]['seed'] and rr[0]['mask_sha256']==rr[1]['mask_sha256']
                        assert rr[2]['seed']!=rr[0]['seed']
                    if unit['arm']=='E1':
                        row=metrics(raw[:,0],raw[:,1],raw[:,2],phi0,phi1)
                        if unit['scenario']=='null':
                            assert np.array_equal(raw[:,0],raw[:,1]) and np.array_equal(phi0,phi1)
                        e1.append({**meta,'method':method,'budget':budget,**row})
                    else:
                        avg=raw.mean(axis=1); a0,a1=phi0.mean(axis=0),phi1.mean(axis=0)
                        row=metrics(avg[:,0],avg[:,1],avg[:,2],a0,a1)
                        e2.append({**meta,'method':method,'budget':budget,**row})
                        windows.append({**meta,'method':method,'budget':budget,
                            'exact':a1-a0,'crn_mean':(avg[:,1]-avg[:,0]).mean(axis=0),
                            'independent_mean':(avg[:,2]-avg[:,0]).mean(axis=0),
                            'v_crn':row['variance_crn'],'v_independent':row['variance_independent'],
                            'same_position_instances':sum(a==b for a,b in zip(doc['extra']['left_indices'],doc['extra']['right_indices']))})
    e1=pd.DataFrame(e1);e2=pd.DataFrame(e2);costs=pd.DataFrame(costs);windows=pd.DataFrame(windows)
    assert len(e1)==1920 and len(e2)==960 and len(costs)==720*768
    components=[]
    for key,g in windows.groupby(['dataset','split','model','scenario','method','budget'],sort=True):
        assert len(g)==config['windows']
        row=dict(zip(['dataset','split','model','scenario','method','budget'],key))
        exact=variance(np.stack(g.exact))
        row.update(v_exact=exact, same_position_instances=int(g.same_position_instances.sum()))
        for arm in ['crn','independent']:
            mc=float(g[f'v_{arm}'].mean())
            row.update({f'v_mc_{arm}':mc,f'v_mean_adjusted_{arm}':variance(np.stack(g[f'{arm}_mean']))-mc/config['window_draws']})
        row['window_share_crn']=ratio(exact,exact+row['v_mc_crn'])
        components.append(row)
    components=pd.DataFrame(components)
    component_summary=components.groupby(['dataset','model','scenario','method','budget'],sort=True).agg(
        split_count=('split','nunique'),v_exact_mean=('v_exact','mean'),
        v_mc_crn_mean=('v_mc_crn','mean'),v_mc_independent_mean=('v_mc_independent','mean'),
        v_mean_adjusted_crn_mean=('v_mean_adjusted_crn','mean'),
        v_mean_adjusted_independent_mean=('v_mean_adjusted_independent','mean'),
        window_share_crn_mean=('window_share_crn','mean')).reset_index()
    assert component_summary.split_count.eq(5).all()
    component_summary['aggregation']='arithmetic means of five split-level components; not a pooled population variance'
    decision=primary(e1,config)
    destination.mkdir(parents=True,exist_ok=False)
    for name,frame in [('e1_cells',e1),('e2_windows',e2),('query_rank_receipts',costs),('e2_components',components),('e2_component_summary',component_summary)]:
        frame.to_csv(destination/f'{name}.csv',index=False)
    atomic(destination/'primary_decision.json',decision)
    atomic(destination/'validation.json',{'status':'PASS','units':len(planned),'e1_cells':len(e1),'e2_windows':len(e2),
        'query_receipts':len(costs),'fingerprint':fp,'source_manifest_sha256':C.digest(work/'SNAPSHOT_MANIFEST.json'),
        'analysis_source_sha256':C.digest(Path(__file__)),'not_independent_recompute':True})
    return decision


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(analyze(args.work,args.config,args.output),indent=2))
