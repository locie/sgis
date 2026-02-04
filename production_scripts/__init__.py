from os import environ

display = environ.get("DISPLAY")
if not display:# No graphical display available
    environ["QT_QPA_PLATFORM"] = "offscreen"
    # cas d'absence de session X (i.e. pas de support Qt)
    # solution: déclarer une variable d'env:
    #     os.environ["QT_QPA_PLATFORM"] = "offscreen"
    
    