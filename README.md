# pr_sd_xxx

Веса модели нужно скачать по ссылке:
https://disk.yandex.ru/d/T1016-ApKcAWJA
и распаковать архив в рабочую директорию проекта.

wget -O new_filename.zip https://example.com/old_filename.zip
------------------------------------------------

Предварительно сделано 2 скрипта:

1. Скрипт для локального запуска из CLI с параметрами
python3 sd_inf_1.py --prompt_json prompt.json --device cuda:1
Этот скрипт при каждом запуске загружает все модели, обрабатывает запрос и выдае ответ. Работает не быстро.

2. Скрипт для доработки
python3 sd_t2i.py --prompt_json prompt.json
Device берет из env. При запуске импортирует sdgenxx, при этом происходит загрузка всех нужных моделей.
Внутри скрипта можно организовать воркеров, которые будут последовательно отрабатывать запросы на уже загруженнной модели.
Видимо, на каждой видеокарте нужно поднимать этот скрипт, а потом распределять запросы между ними.