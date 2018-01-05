import os
import input_emnist_data as inp

cwd=os.getcwd()
path="{}/{}".format(os.getcwd(),'emnist')

dataset=inp.read_data_sets(path,one_hot=True)

print dataset