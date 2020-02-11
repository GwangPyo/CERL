import pickle
import numpy as np
from PIL import Image

if __name__ == "__main__":
    li = []
    for i in range(25):
        with open("img_sample/sample_{}".format(i), "rb") as f:
            ar = pickle.load(f)
            for a in ar:
                li.append(a)
    np.save("sample_array", np.array(li)/255)
    img = np.load("img_sample/sample_array.npy")

