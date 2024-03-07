import json
import os

import cv2
import numpy as np

ep_train = True
ep_eval = 2
directory_path1 = 'C:\\Users\\sambu\\Documents\\Repositories\\CodeBases\\SWiMM_DEEPeR\\logs\\sac\\sac_1\\network'
directory_path2 = 'C:\\Users\\sambu\\Documents\\Repositories\\CodeBases\\SiM_DEEPeR\\Logs\\network\\0'


def iterate_files_and_subdirectories(directory):
    for root, dirs, files in os.walk(directory):
        extension = root.replace(directory, '')[1:]
        root_mirror = os.path.join(directory_path1 if directory == directory_path2 else directory_path2, extension)
        assert os.path.exists(root_mirror), f'{root_mirror} does not exists'

        if len(files) > 0:
            if 'packets_sent' in root_mirror:
                root_mirror = root_mirror.replace('packets_sent', 'packets_received')
            elif 'packets_received' in root_mirror:
                root_mirror = root_mirror.replace('packets_received', 'packets_sent')

            assert os.path.exists(root_mirror), f'{root_mirror} does not exists'

        if root[:-2].endswith('run') and (ep_eval > 0):
            assert len(dirs) == (ep_eval + 1)

        num = 0
        for file in files:
            file_path = os.path.join(root, file)
            file_path_mirror = os.path.join(root_mirror, file)

            assert os.path.exists(file_path_mirror), f'{file_path_mirror} does not exists'

            if os.path.splitext(file_path)[1] == '.json':
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)

                    with open(file_path_mirror, 'r') as json_file_mirror:
                        data_mirror = json.load(json_file_mirror)
                        assert data == data_mirror

            elif os.path.splitext(file_path)[1] == '.jpg':
                im = cv2.imread(file_path, cv2.IMREAD_COLOR)
                im_mirror = cv2.imread(file_path_mirror, cv2.IMREAD_COLOR)

                assert np.array_equal(im, im_mirror), 'The images are not exactly equivalent.'

            num += 1


iterate_files_and_subdirectories(directory_path1)
iterate_files_and_subdirectories(directory_path2)
