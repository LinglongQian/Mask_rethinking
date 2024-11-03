"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .beijing_air import BeijingAir
from .electricity import Electricity
from .ett_h1 import ETT_h1
from .italy_air import ItalyAir
from .pedestrian import Pedestrian
from .pems import PeMS
from .physionet2012 import PhysioNet2012
from .physionet2019 import PhysioNet2019
from .finger_movements import finger_movements
from .NATOPS import NATOPS

HPO_RESULTS = {
    "Pedestrian": Pedestrian,
    "PeMS": PeMS,
    "PhysioNet2012": PhysioNet2012,
    "PhysioNet2019": PhysioNet2019,
    "Electricity": Electricity,
    "ETT_h1": ETT_h1,
    "BeijingAir": BeijingAir,
    "ItalyAir": ItalyAir,
    "finger_movements": finger_movements,
    "NATOPS": NATOPS
}

__all__ = [
    "HPO_RESULTS",
]
