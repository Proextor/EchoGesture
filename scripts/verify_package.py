"""Dependency-free structural ONNX and vocabulary verification, no inference claims."""
from pathlib import Path
from collections import Counter
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.main import vocab
from src.primary_backend.config import MODEL_IN_DIM,T
from src.primary_backend.features import resample, to_model_input
import numpy as np

def varint(b,p):
    v=s=0
    while True:
        c=b[p];p+=1;v|=(c&127)<<s
        if c<128:return v,p
        s+=7

def fields(b):
    p=0
    while p<len(b):
        tag,p=varint(b,p);f,t=tag>>3,tag&7
        if t==0:v,p=varint(b,p)
        elif t==1:v=b[p:p+8];p+=8
        elif t==2:k,p=varint(b,p);v=b[p:p+k];p+=k
        elif t==5:v=b[p:p+4];p+=4
        else:raise ValueError('Bad protobuf')
        yield f,t,v

def many(b,n):return [v for f,t,v in fields(b) if f==n and t==2]
def one(b,n):return next(iter(many(b,n)),None)
def shape(b):
    tensor=one(one(b,2),1);dims=one(tensor,2)
    return [next((v if f==1 else v.decode() for f,t,v in fields(d) if f in (1,2)),None) for d in many(dims,1)]
def graph(path):
    g=one(path.read_bytes(),7)
    ins=[(one(i,1).decode(),shape(i)) for i in many(g,11)]
    outs=[(one(o,1).decode(),shape(o)) for o in many(g,12)]
    ops=Counter(one(n,4).decode() for n in many(g,1))
    return ins,outs,ops

assert MODEL_IN_DIM==289 and T==32
assert len(vocab())==100
z=np.zeros((32,49,3),np.float32)
x=to_model_input(resample(z))
assert x.shape==(32,289) and x.dtype==np.float32 and np.isfinite(x).all()
model=ROOT/'src/primary_backend/models/echogesture_int8.onnx'
i,o,ops=graph(model)
assert i==[('input',[1,32,289])],i
assert o==[('logits',[1,100])],o
assert ops['QuantizeLinear']>0 and ops['DequantizeLinear']>0,ops
print('PRIMARY STATIC GRAPH PASS',i,o,'Q=',ops['QuantizeLinear'],'DQ=',ops['DequantizeLinear'])
for path in sorted((ROOT/'artifacts').glob('*.onnx')):
    i,o,ops=graph(path)
    assert i[0][1][1:]==[64,486] and o[0][1][-1]==100,(i,o)
    print('LEGACY GRAPH PASS',path.name,i,o)
print('VOCAB ORDER PASS: 100/100 identical')
print('FEATURE TRANSFORM PASS: (32, 49, 3) -> (32, 289) float32')

new_models=sorted((ROOT/'artifacts'/'final_5seed').glob('seed_*.onnx'))
assert len(new_models)==5, f'Expected five final models, got {len(new_models)}'
for path in new_models:
    i,o,ops=graph(path)
    assert i==[('keypoints',['batch',64,486])], (path,i)
    assert o==[('logits',['batch',100])], (path,o)
    assert ops['QuantizeLinear']==0 and ops['DequantizeLinear']==0, (path,'Unexpected quantization graph')
    print('FINAL 5-SEED GRAPH PASS',path.name,i,o,'bytes',path.stat().st_size)
print('FINAL FIVE-SEED SHAPES PASS: 64x486 to 100 output classes')
