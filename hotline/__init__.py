
from hotline.hotline import analyze
from hotline.hotline import *

from hotline.annotate import HotlineAnnotate, HotlineAnnotateModule
annotate = HotlineAnnotate().make_wrapper()
annotate_module_list = annotate.annotate_module_list