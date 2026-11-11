"""실제 데이터 받기: Intel Research Lab 2D 라이다 + 오도메트리 (약 16 MB).

출처: 프라이부르크 대학 SLAM 평가 페이지 (Kümmerle et al., 2009)
  http://ais.informatik.uni-freiburg.de/slamevaluation/datasets.php
데이터는 저장소에 넣지 않는다 (data/ 는 .gitignore).
"""
import os
import urllib.request

URL = "http://ais.informatik.uni-freiburg.de/slamevaluation/datasets/intel.clf"
OUT = os.path.join("data", "intel.clf")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    if os.path.exists(OUT):
        print(f"이미 있음: {OUT}")
    else:
        print(f"받는 중: {URL}")
        urllib.request.urlretrieve(URL, OUT)
        print(f"완료: {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")
