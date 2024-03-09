'''Tries to install libs, if not, installs them'''

import importlib
import subprocess

def install(package):
    subprocess.check_call(["pip", "install", package])

def install_libraries():
    try:
        importlib.import_module('os')
        importlib.import_module('numpy')
        importlib.import_module('soundfile')
        importlib.import_module('random')
        importlib.import_module('configparser')
    except ModuleNotFoundError:
        print("Una o più librerie non sono state trovate.")
        risposta = input("Vuoi installare le librerie mancanti? (y/n) ")
        if risposta.lower() == 'y':
            if 'numpy' not in locals():
                install('numpy')
            if 'soundfile' not in locals():
                install('soundfile')
            