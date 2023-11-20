from typing import Type

from myrecall.methods.ewc import EWC_SAC
from myrecall.methods.packnet import PackNet_SAC
from myrecall.methods.clonex import ClonEx_SAC
from myrecall.methods.recall import RECALL_SAC
from myrecall.sac.sac import SAC


def get_sac_class(cl_method: str) -> Type[SAC]:
    if cl_method in ["ft", "pm"]:
        return SAC
    if cl_method == "recall":
        return RECALL_SAC
    if cl_method == "ewc":
        return EWC_SAC
    if cl_method == "packnet":
        return PackNet_SAC
    if cl_method == "clonex":
        return ClonEx_SAC
    assert False, "Bad cl_method!"
