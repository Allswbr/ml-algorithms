import os
import logging
import errno


def mkdir_p(path):
    """http://stackoverflow.com/a/600612/190597 (tzot)"""
    try:
        os.makedirs(path, exist_ok=True)  # Python>3.2
    except TypeError:
        try:
            os.makedirs(path)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # Cоздание обработчика файлов, который регистрирует даже отладочные сообщения
    # Создание консольного обработчика с более высоким уровнем журнала
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # Создание средства форматирования и добавление его в обработчик
    formatter = logging.Formatter(u"%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    # Добавление обработчика в логер
    logger.addHandler(ch)

    return logger


def create_handler(path):
    mkdir_p(os.path.dirname(path))
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(u"%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)

    return fh
