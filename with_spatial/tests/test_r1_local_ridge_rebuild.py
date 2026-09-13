"""Synthetic-data tests only. No real clients, cached pickle, or paper outputs are read."""
from __future__ import annotations
import os
import sys
import unittest
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from experiments.r1_local_ridge_rebuild import (
    ridge_prediction, predict_one_day, prepare_client, validate_frame, align_with_reference,
    _attach_stage_columns, GroupCorrectionConfig, NegativeTransferConfig,
    deployment_group_daily_wape)


def fake_client() -> dict:
    rng = np.random.default_rng(37209)
    n = 85
    x = rng.normal(size=(n, 539))
    x[:, -10:] = np.arange(10)
    baseline = 30 + rng.uniform(0, 20, size=(n, 96))
    residual = rng.normal(size=(n, 96))
    return prepare_client({'station_id': 1009, 'station_type': 'pure_fast',
        'forecast_date': pd.date_range('2020-01-01', periods=n), 'x': x,
        'y_residual': residual, 'baseline': baseline, 'y_true': baseline + residual})


def fake_frame(days: int = 44) -> pd.DataFrame:
    dates = pd.date_range('2023-01-16', periods=days)
    idx = pd.MultiIndex.from_product([['fold_1'], [1009, 1120], dates, range(96)],
        names=['fold','station_id','prediction_date','slot'])
    frame = idx.to_frame(index=False)
    frame['type'] = frame['station_id'].map({1009:'pure_fast',1120:'pure_slow'})
    rng = np.random.default_rng(847)
    frame['y_true'] = rng.uniform(3, 30, size=len(frame))
    frame['previous_day_baseline'] = frame['y_true'] + rng.normal(0,1,size=len(frame))
    frame['prediction'] = frame['y_true'] + rng.normal(0,.6,size=len(frame))
    frame['proposed_prediction'] = frame['y_true'] + rng.normal(0,.3,size=len(frame))
    return frame


class ReconstructionTests(unittest.TestCase):
    def test_dual_equals_centered_primal(self):
        rng = np.random.default_rng(118)
        x = rng.normal(size=(60,539)); r = rng.normal(size=(60,96)); q=rng.normal(size=(2,539))
        x[:,-5:] = np.arange(5); q[:,-5:] = np.arange(5)
        mu=x.mean(0); scale=x.std(0); constant=np.ptp(x,axis=0)==0; scale[constant]=1
        z=(x-mu)/scale; z[:,constant]=0
        for alpha in (100.,1000.):
            got,_ = ridge_prediction(x,r,q,alpha)
            coef=np.linalg.solve(z.T@z+alpha*np.eye(539), z.T@(r-r.mean(0)))
            expected=((q-mu)/scale)@coef+r.mean(0)
            np.testing.assert_allclose(got,expected,rtol=2e-10,atol=2e-10)

    def test_constant_columns_and_intercept(self):
        x=np.tile(np.array([1.,.3,1e8]),(60,1));q=x[[0]].copy()
        r=np.tile(np.arange(96),(60,1)).astype(float)
        q[:,2] += 1e6
        p,info=ridge_prediction(x,r,q,100.)
        np.testing.assert_allclose(p,r[[0]],rtol=0,atol=1e-12)
        self.assertEqual(info['effective_feature_count'],0)

    def test_current_future_labels_not_used(self):
        c=fake_client();d=c['dates'][65]
        before,audit=predict_one_day(c,d,100.)
        c['residual'][65:] += 1e8; c['y_true'][65:] -= 1e7
        c['x'][66:] *= 1000; c['baseline'][66:] += 1e9
        after,_=predict_one_day(c,d,100.)
        np.testing.assert_array_equal(before,after)
        self.assertEqual(audit['history_days'],60)
        self.assertTrue(audit['max_history_date_before_forecast'])
        self.assertEqual(audit['history_last'],str((d-pd.Timedelta(days=1)).date()))

    def test_incomplete_calendar_history_rejected(self):
        c=fake_client()
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            predict_one_day(c,c['dates'][40],100.)
        for k in ('dates','x','residual','baseline','y_true'):
            c[k]=c[k].delete(20) if k=='dates' else np.delete(c[k],20,axis=0)
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            predict_one_day(c,pd.Timestamp('2020-03-06'),100.)

    def test_nonfinite_and_invalid_alpha_rejected(self):
        x=np.ones((60,4));r=np.zeros((60,2));q=x[[0]].copy()
        for a in (0.,-1.,np.nan,np.inf):
            with self.assertRaises(ValueError): ridge_prediction(x,r,q,a)
        q[0,0]=np.nan
        with self.assertRaises(ValueError): ridge_prediction(x,r,q,100.)

    def test_complete_grid_and_duplicate_missing_rejected(self):
        f=fake_frame();ds=pd.date_range('2023-01-16',periods=44)
        validate_frame(f,ds,{'fold_1':[1009,1120]})
        with self.assertRaises(ValueError): validate_frame(f.iloc[:-1],ds,{'fold_1':[1009,1120]})
        with self.assertRaises(ValueError): validate_frame(pd.concat([f,f.iloc[[0]]]),ds,{'fold_1':[1009,1120]})

    def test_key_alignment_and_truth_check(self):
        f=fake_frame(); reference=f.drop(columns=['prediction'])
        shuffled=f.drop(columns=['proposed_prediction']).sample(frac=1,random_state=4)
        aligned,diffs=align_with_reference(shuffled,reference)
        self.assertEqual(len(aligned),len(f));self.assertEqual(diffs['y_true'],0)
        shuffled.loc[shuffled.index[0],'y_true']+=.01
        with self.assertRaises(ValueError): align_with_reference(shuffled,reference)

    def test_nonnegative_output(self):
        c=fake_client();c['residual'][:]=-100.;c['baseline'][:]=0
        p,_=predict_one_day(c,c['dates'][65],100.)
        np.testing.assert_array_equal(p,np.zeros(96))

    def test_original_fairness_helper_is_causal(self):
        f=fake_frame();gc=GroupCorrectionConfig();nc=NegativeTransferConfig()
        first=_attach_stage_columns(f,'test','prediction',gc,nc)
        g=f.copy(); cut=pd.Timestamp('2023-02-05')
        # A new outcome on/after cut cannot change cut's deployed prediction.
        g.loc[g.prediction_date>=cut,'y_true']+=1000
        second=_attach_stage_columns(g,'test','prediction',gc,nc)
        cols=['r1_test_group','r1_test_group_risk','r1_test_final_enabled']
        np.testing.assert_array_equal(first.loc[f.prediction_date<=cut,cols].values,
                                      second.loc[g.prediction_date<=cut,cols].values)

    def test_daily_metric_does_not_cancel_across_folds(self):
        # Opposite errors in distinct deployment groups must not cancel.
        f=fake_frame(days=1)
        g=f.copy();g['fold']='fold_2';g['station_id']+=10000
        f['y_true']=10.;f['prediction']=12.
        g['y_true']=10.;g['prediction']=8.
        value=deployment_group_daily_wape(pd.concat([f,g]),'prediction').iloc[0]
        self.assertAlmostEqual(value,.2)


if __name__=='__main__':
    print('SYNTHETIC_TESTS_ONLY; REAL_DATA_TRAINING=NO',flush=True)
    unittest.main(verbosity=2)
