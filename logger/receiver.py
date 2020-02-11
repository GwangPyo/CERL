from logger.meta import BINARY_DATA_PATH
import scp
import paramiko
import pickle


class Receiver(object):
    HOST_IP = {
        "DEEPONE": "115.145.178.111",
        "YKSI": "115.145.178.112"
    }
    def __init__(self, host,  user="yoo"):
        self.user = user
        self.host = host
        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            with open("ps.pkl", "rb") as f:
                pwd = pickle.load(f)
                pwd = pwd.decode("UTF-8")
        except FileNotFoundError:
            print("no key has found")
            pwd = input("pwd")
        self.ssh_client.connect(hostname=host, username=user, password=pwd)

    def flush(self, key):
        target_path = BINARY_DATA_PATH + "/" + key
        print(target_path)
        with scp.SCPClient(self.ssh_client.get_transport()) as cli:
            cli.get(target_path, BINARY_DATA_PATH, preserve_times=True, recursive=True)

        self.ssh_client.exec_command("rm -r " + target_path)
        self.ssh_client.exec_command("mkdir " + target_path)

    def pull_csv_only(self, key):
        target_path = BINARY_DATA_PATH + "/" + key + "/temp.csv"
        with scp.SCPClient(self.ssh_client.get_transport()) as cli:
            cli.get(target_path, BINARY_DATA_PATH, preserve_times=True, recursive=False)

        self.ssh_client.exec_command("rm " + target_path)

    def __del__(self):
        self.ssh_client.close()



if __name__ == "__main__":
    receiver = Receiver("115.145.178.111")
    receiver.pull_csv_only("DEEPONE")