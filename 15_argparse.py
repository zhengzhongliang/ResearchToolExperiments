import argparse


def main():


    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument("--bs", type = int, default=5)
    parser.add_argument("--device", type = str, default="cpu")

    args = parser.parse_args()
    print(args.bs)
    print(args.device)

    return 0

main()