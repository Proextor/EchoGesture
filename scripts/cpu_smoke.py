"""Real CPU ONNX inference test. Optional MP4 argument tests full video path."""
from pathlib import Path
import sys,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.main import primary_probs_from_features,legacy_probs_from_features,final_probs_from_features,predict,vocab
f=final_probs_from_features(np.zeros((64,486),np.float32))
assert f.shape==(100,) and np.isclose(f.sum(),1) and np.isfinite(f).all()
print('FINAL FIVE-SEED ONNX CPU PASS: (1,64,486) -> (1,100)')
b=primary_probs_from_features(np.zeros((32,289),np.float32))
assert b.shape==(100,) and np.isclose(b.sum(),1) and np.isfinite(b).all()
print('PRIMARY ONNX CPU PASS: (1,32,289) -> (1,100)')
o=legacy_probs_from_features(np.zeros((64,486),np.float32))
assert o.shape==(100,) and np.isclose(o.sum(),1) and np.isfinite(o).all()
print('LEGACY THREE-SEED CPU PASS: (1,64,486) -> (1,100)')
if len(sys.argv)>1:
    r=predict(sys.argv[1]);assert r['prediction'] in vocab() and len(r['top5'])==5
    print('FULL VIDEO PREDICT PASS:',r)
else:
    print('Full video not tested: provide path to a real video')
