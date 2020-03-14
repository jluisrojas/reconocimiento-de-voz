from preparacion import *

distribuciones = ["test"]

def main():
    for distribucion in distribuciones:
        common_proc_distrib("common-voice/es/", distribucion+".tsv", "clips/")

if __name__ == "__main__":
    main()
