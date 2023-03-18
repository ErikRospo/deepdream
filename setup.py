# needs to download to ./models/
#http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
import os
import requests
import hashlib

def hash_and_compare(path, expected_hash):
    with open(path, 'rb') as f:
        file_contents = f.read()
    sha1_hash = hashlib.sha1(file_contents).hexdigest()
    return sha1_hash == expected_hash
if not os.path.exists("models"):
    os.mkdir("models")
    print("downloading the model")
    with open("models/bvlc_googlenet.caffemodel", 'wb') as fd:
        r=requests.get("http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel",stream=True)
        print(r.headers)
        i=0
        maxsize=r.headers.get("content-length")
        cs=4096
        for chunk in r.iter_content(chunk_size=cs):
            i+=cs
            print(f"({i}/{maxsize}): {round(100*(i/int(maxsize)),2)}%")
            fd.write(chunk)
    matches=hash_and_compare("models/bvlc_googlenet.caffemodel","405fc5acd08a3bb12de8ee5e23a96bec22f08204")
    if matches:
        print("Model verified")
    else:
        print("Model didn't download correctly. Please try again")
        exit()
    print("model downloaded")
    print("downloading deploy.prototxt")
    with open("models/deploy.prototxt","wb") as f:
        a=requests.get("https://github.com/BVLC/caffe/raw/master/models/bvlc_googlenet/deploy.prototxt")
        f.write(a.content)
    print("prototxt downloaded")
else:
    print("Model Directory already exists, not overwriting anything")