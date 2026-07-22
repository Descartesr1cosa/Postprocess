"""Top-level switchboard for MPCNS post-processing.

Change only this file for normal use, then run ``python run_post.py``.
It reads/calculates once through post_core.py and chooses output modules.
"""

from pathlib import Path

from post_tecplot import export_node_tecplot
from post_na_altitude import export_na_altitude_profiles
from post_virtual_flight import export_virtual_flight
import runpy


# ============================================================
# User settings
# ============================================================
# Change this to the directory that contains your DATA and DATA_bin folders.
CASE_DIR = Path(r"E:\path\to\your\MPCNS\output")

# Main fluid / electromagnetic Tecplot Node files.
RUN_NODE_TECPLT = False
# dawn.dat, subsolar.dat and dusk.dat Na+ altitude profiles.
RUN_NA_ALTITUDE_PROFILES = True
# MESSENGER trajectory sampled from the nearest simulation Cell.
RUN_VIRTUAL_FLIGHT = True
# Your current DATA/DATA_bin are debug outputs.  Enable this to compare the
# reconstructed DEC current with the solver's saved edge current.
VALIDATE_DEC_WITH_DEBUG_JEDGE = False


if __name__ == "__main__":
    # The expensive read/calculation/Node interpolation happens only once.
    data = runpy.run_path(
        Path(__file__).with_name("post_core.py"),
        init_globals={
            "CASE_DIR": CASE_DIR,
            "VALIDATE_DEC_WITH_DEBUG_JEDGE": VALIDATE_DEC_WITH_DEBUG_JEDGE,
        },
    )

    if RUN_NODE_TECPLT:
        export_node_tecplot(data)
    if RUN_NA_ALTITUDE_PROFILES:
        export_na_altitude_profiles(data)
    if RUN_VIRTUAL_FLIGHT:
        export_virtual_flight(data)
