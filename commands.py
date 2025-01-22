import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Запускает процесс обучени/тестировани/инференса \
            обученной pytorch модели"
    )

    sub_p = parser.add_subparsers(dest="task")

    train_parser = sub_p.add_parser("train", help="Запуск обучения (train.py)")
    test_parser = sub_p.add_parser("test", help="Запуск валидации (test.py)")
    inf_parser = sub_p.add_parser("infer", help="Запуск инференса (infer.py)")

    train_parser.add_argument(
        "aux_args",
        type=str,
        default=None,
        nargs="*",
        metavar="",
        help="Изменение дефолтных аргументов для трейна (train_config)",
    )

    inf_parser.add_argument(
        "image_path",
        type=str,
        default=None,
        nargs="*",
        metavar="",
        help="Изображение для инференса в формате .JPG (infer_config.file_path)",
    )

    args = parser.parse_args()

    # немного замудрено, но требования в задании именно использовать
    # commands.py. В продакшене лучше просто обойтись poetry poe
    if args.task == "train":
        subprocess.run(["poetry", "poe", "train"] + args.aux_args)
    elif args.task == "test":
        subprocess.run(["poetry", "poe", "test"])
    elif args.task == "infer":
        subprocess.run(["poetry", "poe", "infer"] + args.image_path)
