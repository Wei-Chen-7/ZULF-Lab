#!/bin/sh
# Regenerate every reported result, in dependency order, then re-render every
# document from the store. Each script records the numbers it owns into
# results.json; nothing is transcribed by hand. See results.py.
#
# Networks are cached in models/, so a second run only re-evaluates. From a
# clean checkout the first run trains five networks and takes a few hours.
#
# make_figure1.py is not here: it reads the published PDF of ref [1], which is
# not redistributable, so it takes the path as an argument.
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
echo "=== render documents ==="   && python -u results.py
echo "=== demo page ==="          && python -u make_demo_page.py
echo "=== ALL DONE ==="
