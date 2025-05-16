# Example TRex Stateless traffic profile (Python API)
from trex_stl_lib.api import *

def create_stream():
    base_pkt = Ether()/IP(src="10.0.0.1", dst="10.0.0.2")/UDP()/Raw(load='X'*64)
    return STLStream(
        packet=STLPktBuilder(pkt=base_pkt),
        mode=STLTXCont(percentage=10)
    )

def register():
    return [create_stream()]

