import os

from model_runs.src.create_runs import servers


if __name__ == "__main__":
    """ Using the server list from create runs, make a nodes.txt file.
    """

    txt = ""
    for server in servers.keys():
        txt += server + "\n"
    try:
        os.remove("scripts/nodes.txt")
    except Exception as E:
        print(E)

    with open("scripts/nodes.txt", "a") as handle:
        handle.write(txt)
