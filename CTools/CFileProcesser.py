import os
import numpy as np
import json

class CFileProcesser:
    def get_filenames(self,path):
        filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(path)) for f in fn]
        filenames.sort()
        return filenames

    def check_file_or_path(self,filename):
        if os.path.isfile(filename):
            print("valid filename: ",filename)
            return True
        elif os.path.isdir(filename):
            print("valid path: ",filename)
            return True
        else:
            print("invalid filename or path: ",filename)
            return False

    def print_filenames(self,path):
        print('*' * 80)
        for file in path:
            print(file)

    def read_sustech_json(self,json_file):
        with open(json_file) as js:
            js_content = json.load(js)
        # print(js_content)
        centers = []
        rotations = []
        scales = []
        for i in range(len(js_content)):
            obj_tmp = js_content[i]
            position = obj_tmp["psr"]["position"]
            rotation = obj_tmp["psr"]["rotation"]
            scale = obj_tmp["psr"]["scale"]

            center_xyz = np.array(list(position.items()))[:, 1].astype(np.float64)
            rotation_xyz = np.array(list(rotation.items()))[:, 1].astype(np.float64)
            scale_xyz = np.array(list(scale.items()))[:, 1].astype(np.float64)

            centers.append(center_xyz)
            rotations.append(rotation_xyz)
            scales.append(scale_xyz)
        return centers, rotations, scales

    def get_file_ids(self,file_names):
        ids = []
        for i in range(len(file_names)):
            file = file_names[i]
            id = file.split("/")[-1].split('.')[0]
            # print(1)
            ids.append(id)
        return ids

    def extract_inds(self,small_ids,big_ids):
        small_ids = np.asarray(small_ids, dtype=int)
        big_ids = np.asarray(big_ids, dtype=int)

        ind_all = []
        for id in small_ids:
            ind = np.argwhere(big_ids == id)
            ind_all.append(ind)
        ind_all = np.asarray(ind_all).reshape(-1)
        return ind_all
