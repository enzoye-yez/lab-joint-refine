import os
from pathlib import Path
import sys

if not getattr(sys, '_harness_runtime', None):
    os.execv(sys.executable, [sys.executable, '-B', '-P',
        str(Path(__file__).resolve().parents[1]/'runtime_bootstrap.py'),
        '--module', 'scene_generation', '--', *sys.argv[1:]])

from .cli import main

raise SystemExit(main())
