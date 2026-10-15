print("[BridgeReceiver] __init__.py loading ...")

import traceback

WEB_DIRECTORY = "./js"

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    print(f"[BridgeReceiver] __init__.py OK - found nodes: {list(NODE_CLASS_MAPPINGS.keys())}")
except Exception as e:
    print(f"[BridgeReceiver] ERROR loading nodes.py: {e}")
    traceback.print_exc()
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
