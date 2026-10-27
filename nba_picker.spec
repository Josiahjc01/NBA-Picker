# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Windows NBA-Picker.exe (run on Windows via build_windows.bat)
# Does NOT produce a Linux binary — build on a Windows machine only.

block_cipher = None

a = Analysis(
    ['gui.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('.env.example', '.'),
        ('README.md', '.'),
        ('data/static_arenas.py', 'data'),
    ],
    hiddenimports=[
        'main',
        'config',
        'data.fetch',
        'data.process',
        'data.espn',
        'data.cache_util',
        'data.static_arenas',
        'model.predict',
        'model.brief',
        'nba_api',
        'pandas',
        'numpy',
        'bs4',
        'lxml',
        'dotenv',
        'requests',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='NBA-Picker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # windowed GUI — not a console-only app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
