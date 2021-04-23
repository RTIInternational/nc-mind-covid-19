import os
from collections import defaultdict
from model_runs.src.create_runs import servers

RUN_PATH = os.environ["RUN_PATH"]


def copy_runs(directories: int, host: str):
    if host == ":":
        # Special GNU parallel value representing localhost
        dest_loc = RUN_PATH
    else:
        dest_loc = f"{host}:{RUN_PATH}"

    os.system("rsync -r {} {}/model_runs/parallel_runs/".format(directories, dest_loc))


if __name__ == "__main__":

    with open("scripts/nodes.txt", "r") as f:
        x = f.readlines()

    # Create a list of server names. This will ensure the correct distribution of runs makes it to each server.
    use_servers = []
    for server in x:
        if "#" not in server:
            server = server.replace("\n", "")
            use_servers.extend([server for s in range(servers[server])])

    # How many runs are there?
    runs = [item for item in os.listdir("model_runs/parallel_runs") if "run_" in item]
    # Duplicate the server names enough times.
    while len(use_servers) < len(runs):
        use_servers.extend(use_servers)

    # Create a string of all the directories to copy to each server
    copy_dict = defaultdict(str)
    for i, run in enumerate(runs):
        copy_dict[use_servers[i]] += "model_runs/parallel_runs/{} ".format(run)

    # Copy the directories to the servers
    for k, v in copy_dict.items():
        copy_runs(v, k)
