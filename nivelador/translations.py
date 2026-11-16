"""Loading of the translation files (.qm) reviewed with Qt Linguist.

The texts of the program are written in English and wrapped in ``tr()``. Every
language lives in ``nivelador/i18n/nivelador_<code>.ts`` (editable with Qt
Linguist) and the program loads its compiled ``.qm`` version.

Workflow (run from the folder that contains the ``nivelador`` package):

    # 1) Extract new or changed texts into the .ts files
    pylupdate6 nivelador/*.py -ts nivelador/i18n/nivelador_es.ts

    # 2) Review and translate
    linguist nivelador/i18n/nivelador_es.ts

    # 3) Compile to .qm (in Linguist this is "File > Release")
    lrelease nivelador/i18n/nivelador_es.ts

Any language can be added by creating its own ``nivelador_<code>.ts`` file; it
shows up in the language menu as soon as its ``.qm`` exists.
"""

from __future__ import annotations

import os

from PyQt6 import QtCore

SOURCE = "en"                     # language the source code is written in
DOMAIN = "nivelador"              # base name of the translation files
I18N_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "i18n")

# Name of each language in its own tongue (never translated).
NAMES = {
    "en": "English",
    "es": "Español",
    "pt": "Português",
    "fr": "Français",
    "de": "Deutsch",
    "it": "Italiano",
}

_current = SOURCE


def available() -> list[tuple[str, str]]:
    """Available languages: (code, display name), the original one first."""
    out = [(SOURCE, NAMES[SOURCE])]
    if os.path.isdir(I18N_DIR):
        prefix = DOMAIN + "_"
        for name in sorted(os.listdir(I18N_DIR)):
            if not name.startswith(prefix) or not name.endswith(".qm"):
                continue
            code = name[len(prefix):-3]
            out.append((code, NAMES.get(code, code)))
    return out


def current() -> str:
    """Code of the language currently installed."""
    return _current


def preferred() -> str:
    """Language to use at startup: the system one if translated, else the original."""
    base = QtCore.QLocale.system().name().split("_")[0]
    return base if any(code == base for code, _ in available()) else SOURCE


def install(app: QtCore.QCoreApplication, code: str) -> None:
    """Installs the requested language (``SOURCE`` keeps the original texts)."""
    global _current
    for old in app.findChildren(QtCore.QTranslator):
        app.removeTranslator(old)
        old.deleteLater()
    if code != SOURCE:
        translator = QtCore.QTranslator(app)
        if translator.load(os.path.join(I18N_DIR, "%s_%s.qm" % (DOMAIN, code))):
            app.installTranslator(translator)
    _current = code
