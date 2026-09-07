#!/bin/sh
# Regenerate every reported result, in dependency order.
#
# Networks are cached in models/, so a second run only re-evaluates. From a
# clean checkout the first run trains five networks and takes a few hours;
# tighten_proposal.py is the long pole because the wide-flow configuration is
# trained only for that comparison.
#
# make_figure1.py is deliberately not here: it reads the published PDF of
# ref [1], which is not redistributable, so it takes the path as an argument.
set -e
cd "$(dirname "$0")"
echo "=== nested_reference ==="   && python -u nested_reference.py
echo "=== train_library ==="      && python -u train_library.py
echo "=== local_baseline ==="     && python -u local_baseline.py
echo "=== methanol_demo ==="      && python -u methanol_demo.py
echo "=== resolution_cliffs ===" && python -u resolution_cliffs.py
echo "=== final_check ==="        && python -u final_check.py
echo "=== tighten_proposal ==="   && python -u tighten_proposal.py
echo "=== figure 2 ==="           && python -u make_figure2.py
echo "=== figure 4 ==="           && python -u make_figure4.py
echo "=== demo page ==="          && python -u make_demo_page.py
echo "=== ALL DONE ==="
