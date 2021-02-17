'''
@File   :   global_var.py
@Time   :   2021-1-2 
@Author :   wang
'''
class global_var:
    data_path = None
    class_path = None

def set_data_path(data_path):
    global_var.data_path = data_path
def get_data_path():
    return global_var.data_path
def set_class_path(class_path):
    global_var.class_path = class_path
def get_class_path():
    return global_var.class_path
