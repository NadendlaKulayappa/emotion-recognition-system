import importlib

modules = [
    "cv2",
    "torch",
    "numpy",
    "PIL",
    "torchvision",
    "transformers",
    "groq",
]

def get_version(mod_name):
    try:
        mod = importlib.import_module(mod_name)
        if hasattr(mod, '__version__'):
            return mod.__version__
        elif mod_name == "PIL":
            from PIL import Image
            return Image.__version__
        elif mod_name == "cv2":
            import cv2
            return cv2.__version__
        else:
            return "Version info not available"
    except Exception as e:
        return f"Not installed ({e})"

print("📦 Installed Module Versions:")
print("=" * 40)
for m in modules:
    print(f"{m:<20}: {get_version(m)}")
