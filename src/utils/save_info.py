import os
import json
from src.configuration_file import *


def save_info(info, info_type, name):
    results_path = os.getcwd() + "/results/{}/{}/".format(EXPERIMENT_NAME, info_type)
    try:
        os.makedirs(results_path)
    except OSError:
        pass

    file_name = {
        "dataframe": ".csv",
        "figure": ".png",
        "json": ".json",
        "text": ".txt",
    }
    name = results_path + name + file_name[info_type]
    print(f"Saving {name}")
    if info_type == "dataframe":
        info.to_csv(path_or_buf=name)
    elif info_type == "figure":
        info.savefig(name)
    elif info_type == "json":
        with open(name, "w") as outfile:
            json.dump(info, outfile)
        outfile.close()
    elif info_type == "text":
        info += '\n'
        try:
            outfile = open(name, "r")
            content = outfile.readlines()
            content.append(info)
        except:
            content = info
        outfile = open(name, "w")
        outfile.writelines(content)
        outfile.close()
    return
