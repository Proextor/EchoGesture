"""Official root-level predict(video_path) API."""
from src.main import predict
if __name__ == '__main__':
    import json,sys
    if len(sys.argv)!=2:raise SystemExit('Usage: python predict.py path/to/video.mp4')
    print(json.dumps(predict(sys.argv[1])))
