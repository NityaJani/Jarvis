"""py2app build script for the Jarvis menu-bar app.

Build (alias mode - references the existing venv/source instead of bundling
everything, so edits to jarvis/*.py take effect without rebuilding):

    ./venv/bin/python setup.py py2app -A
"""

from setuptools import setup

APP = ["jarvis/tray_app.py"]
OPTIONS = {
    "argv_emulation": False,
    "iconfile": "AppIcon.icns",
    "plist": {
        "CFBundleName": "Jarvis",
        "CFBundleDisplayName": "Jarvis",
        "CFBundleIdentifier": "local.jarvis",
        "NSMicrophoneUsageDescription": (
            "Jarvis listens for its wake word and your spoken commands."
        ),
    },
}

setup(
    app=APP,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
