import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Windows setup: keep all bank data under the checked-in Banks/ folder.
BANKS_ROOT = os.getenv("BANKS_DIR", os.path.join(PROJECT_ROOT, "graph_database", "data", "Banks"))

# macOS setup (commented for now):
# BANKS_ROOT = os.getenv("BANKS_DIR", os.path.join(PROJECT_ROOT, "Banks"))


def get_bank_path(bank_name):
    bank_folders = {
        "banka": "Bank_A",
        "bankb": "Bank_B",
        "bankc": "Bank_C",
    }
    folder_name = bank_folders[bank_name.lower()]
    return os.path.join(BANKS_ROOT, folder_name)


def get_bank_file(bank_name, filename):
    return os.path.join(get_bank_path(bank_name), filename)
