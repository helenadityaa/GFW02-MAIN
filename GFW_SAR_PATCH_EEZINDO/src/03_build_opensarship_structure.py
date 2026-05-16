from utils import REQUIRED_DIRECTORIES, ensure_directories


def main():
    ensure_directories()
    print("Struktur folder OpenSARShip-like siap:")
    for path in REQUIRED_DIRECTORIES:
        print(f"- {path}")


if __name__ == "__main__":
    main()
