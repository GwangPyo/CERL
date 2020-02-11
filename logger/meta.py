import os
import pickle

HOME = os.path.expanduser("~")
BINARY_DATA_PATH = HOME + "/data"


def CREATE_SAVE_PATH():
    try:
        os.mkdir(BINARY_DATA_PATH)
    except FileExistsError:
        print("there is already saving path" + BINARY_DATA_PATH)
    return


def GET_METADATA():
    cwd = os.path.abspath(os.path.curdir)
    try:
        with open(cwd + "/logger/columns", "rb") as f:
            col = pickle.load(f)
    except FileNotFoundError:
        print("No column has been found. exit")
        exit(1)
    return col


def SET_METADATA(LIST):
    with open("columns", "wb") as f:
        pickle.dump(LIST, f)


class MACHINE_KEY(object):
    def __init__(self):
        """
        Object for obtain key
        """
        self.k = self.get_key()

    @staticmethod
    def get_key()->str:
        try:
            with open(BINARY_DATA_PATH + "/key.txt", "r") as f:
                k = f.readline()

        except FileNotFoundError:
            k = input("There is no key in your current machine. Input your key for the prefix of your machine\n")
            with open(BINARY_DATA_PATH + "/key.txt", "w") as f:
                f.writelines(k)
            os.mkdir(BINARY_DATA_PATH + "/" + k)

        return k

    @staticmethod
    def set_key():
        k = input("Input your key for the prefix of your machine\n")
        with open(BINARY_DATA_PATH + "/key.txt", "w") as f:
            f.writelines(k)
            print("write key ")
        os.mkdir(BINARY_DATA_PATH + "/" + k)

    def __call__(self, *args, **kwargs):
        return self.k


def get_sample_number(sample_number=None):
    cwd = os.path.abspath(os.path.curdir)
    sample_number_path = cwd + "/logger/sample_number"
    if sample_number is None:
        try:
            with open(sample_number_path, "r") as f:
                sample_number = f.readline()

        except FileNotFoundError:
            print("Sample number file Not found. Please give argument as a sample number")
            exit(-1)

        with open(sample_number_path, "w") as f:
            sample_number = int(sample_number) + 1
            f.writelines(str(sample_number))

    else:
        with open(sample_number_path, "w") as f:
            f.writelines(str(sample_number))
    return int(sample_number)


def get_model_dir(key, sample_number=None):
    id_ = get_sample_number(sample_number)
    if key is None:
        key = get_key()
    path = BINARY_DATA_PATH + "/" + key + "/" + str(id_)
    try:
        os.mkdir(path)
    except FileExistsError:
        print("you have wrong sample number, give argument as a sample number to fix")
        exit(-1)
    return path


def write_temp(key, log_values):
    # write temp file first, hold it into dir
    # so that write full log after learning and evaluation finished
    if key is None:
        key = get_key()
    temp = BINARY_DATA_PATH + "/" + key + "/temp.csv"
    col = GET_METADATA()
    if not os.path.isfile(temp):

        with open(temp, "w") as f:
            for c in col[1:-1]:
                f.write(c + ",")
            f.write(col[-1])
            f.write("\n")

    with open(temp, "a") as f:
        for c in col[1:-1]:
            f.write(str(log_values[c]) + ",")
        f.write(str(log_values[col[-1]]))
        f.write("\n")


def get_key():
    return MACHINE_KEY.get_key()


def INIT():
    CREATE_SAVE_PATH()
    MACHINE_KEY()
    return

if __name__=="__main__":
    print(MACHINE_KEY.set_key())