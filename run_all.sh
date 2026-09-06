#!/bin/sh
# Everything downstream of the trained networks, in dependency order.
# Networks are cached in models/, so only train_library.py trains anything new.
set -e
cd "$(dirname "$0")"
echo "=== train_library ==="     && python -u train_library.py
echo "=== local_baseline ==="    && python -u local_baseline.py
echo "=== methanol_demo ==="     && python -u methanol_demo.py
echo "=== resolution_cliffs ===" && python -u resolution_cliffs.py
echo "=== final_check ==="       && python -u final_check.py
echo "=== figure 2 ==="          && python -u make_figure2.py
echo "=== figure 4 ==="          && python -u make_figure4.py
echo "=== ALL DONE ==="
