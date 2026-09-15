import gzip, struct, urllib.request
from pathlib import Path

URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
OUT = Path(__file__).resolve().parent.parent / "MNIST"
OUT.mkdir(parents=True, exist_ok=True)

def get(name):
    return gzip.decompress(urllib.request.urlopen(URL + name).read())


for csv, img, lbl in [("mnist_train.csv", "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"), 
                      ("mnist_test.csv", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")]:
    if (OUT / csv).exists():
        print(f"The {OUT / csv} exists, so continue.\n")
        continue
    pixels, labels = get(img)[16:], get(lbl)[8:]
    with (OUT / csv).open("w") as f:
        for i, label in enumerate(labels):
            f.write(f"{label}," + ",".join(map(str, pixels[i * 784 : (i + 1) * 784])) + "\n")
    