"""Small deterministic convex heads; all scaling is fitted on training rows only."""
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from .contracts import require,protocol

def standardize_fit(x):
    x=np.asarray(x,dtype=np.float64);require(x.ndim==2 and np.isfinite(x).all(),'Invalid training matrix')
    mean=x.mean(0);std=x.std(0);active=std>1e-12;scale=np.where(active,std,1.)
    return mean,scale,active

def objective(theta,x,y,l2):
    w,b=theta[:-1],theta[-1];z=x@w+b
    value=np.mean(np.logaddexp(0,z)-y*z)+.5*l2*np.dot(w,w)
    error=(expit(z)-y)/len(y)
    gradient=np.r_[x.T@error+l2*w,error.sum()]
    return float(value),gradient

def fit(x,y,l2,settings=None):
    settings=settings or protocol()['solver'];y=np.asarray(y,dtype=np.float64)
    require(y.shape==(len(x),) and set(y)=={0.,1.},'Training labels must have both classes')
    require(l2>0,'L2 penalty must be positive')
    mean,scale,active=standardize_fit(x);xx=np.ascontiguousarray(((x-mean)/scale)[:,active])
    initial=np.zeros(xx.shape[1]+1);initial[-1]=np.log(y.mean()/(1-y.mean()))
    attempts=[];result=None
    for maxiter,ftol in [(settings['maxiter'],settings['ftol']),(settings['retry_maxiter'],1e-15)]:
        result=minimize(objective,initial,args=(xx,y,float(l2)),jac=True,method='L-BFGS-B',
            options=dict(maxiter=maxiter,ftol=ftol,gtol=settings['gtol'],maxls=50,maxcor=20))
        value,grad=objective(result.x,xx,y,float(l2));g=float(np.max(np.abs(grad)))
        attempts.append(dict(iterations=int(result.nit),evaluations=int(result.nfev),success=bool(result.success),
            message=str(result.message),objective=value,gradient_inf=g))
        if np.isfinite(value) and np.isfinite(result.x).all() and g<=settings['acceptable_gradient_inf']:break
        initial=result.x
    else:raise RuntimeError('Logistic solver failed stationarity check: '+str(attempts))
    w=np.zeros(len(mean));w[active]=result.x[:-1]
    model=dict(weight=w,intercept=float(result.x[-1]),mean=mean,scale=scale,active=active,l2=float(l2))
    diagnostics=dict(attempts=attempts,active_features=int(active.sum()),training_examples=len(y),l2=float(l2))
    return model,diagnostics

def predict(model,x):
    x=np.asarray(x,dtype=np.float64);require(x.shape[1]==len(model['weight']) and np.isfinite(x).all(),'Prediction dimensions or values invalid')
    return (((x-model['mean'])/model['scale'])*model['active'])@model['weight']+model['intercept']

def select(x_fit,y_fit,x_val,y_val,settings=None):
    settings=settings or protocol()['solver'];trials=[];winner=None
    y_val=np.asarray(y_val,dtype=np.float64)
    require(set(y_val)=={0.,1.},'Inner validation labels must have both classes')
    for l2 in settings['l2_grid']:
        model,diagnostics=fit(x_fit,y_fit,l2,settings);z=predict(model,x_val)
        loss=float(np.mean(np.logaddexp(0,z)-y_val*z));trials.append(dict(l2=float(l2),inner_bce=loss,solver=diagnostics))
        if winner is None or loss<winner['inner_bce']-1e-12 or (abs(loss-winner['inner_bce'])<=1e-12 and l2>winner['l2']):winner=trials[-1]
    return winner['l2'],trials

def save_model(path,model):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**model)
    tmp.replace(path)

def load_model(path):
    with np.load(path,allow_pickle=False) as z:m={k:z[k].copy() for k in z.files}
    require(set(m)=={'weight','intercept','mean','scale','active','l2'},'Unexpected saved model schema')
    m['intercept']=float(m['intercept']);m['l2']=float(m['l2']);m['active']=m['active'].astype(bool)
    require(np.isfinite(m['weight']).all() and np.isfinite(m['mean']).all() and np.isfinite(m['scale']).all() and np.all(m['scale']>0) and np.isfinite(m['intercept']),'Nonfinite saved model')
    require(np.all(m['weight'][~m['active']]==0),'Disabled feature has a nonzero weight')
    return m
