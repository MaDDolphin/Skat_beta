# Подключаем модуль
import os
import yaml

import matplotlib.pyplot as plt
from random import randint

# Получаем список файлов в переменную files
files = os.listdir('./')
big_data = []
for file in files:
    if file.endswith('.yaml'):

        meta = (file.split('_'))
        meta.pop()
        meta[0] = ('.'.join(meta[0].split('.')[:-1]))
        with open(file, 'r') as stream:
            data = yaml.safe_load(stream)

            cpu = [0.0]
            fps = [0.0]
            gpu = [0.0]
            ram = [0.0]
            frames = [0.0]
            count = 0
            for frame in data:
                count = count + 1
                cpu.append(frame['cpu'])
                fps.append(frame['fps'])
                gpu.append(frame['gpuload']* 100)
                ram.append(frame['ram'])
                frames.append(count)

        meta.append(cpu)
        meta.append(fps)
        meta.append(gpu)
        meta.append(ram)
        meta.append(frames)
        big_data.append(meta)


fig, ax = plt.subplots()
for data in big_data:
    ax.plot(data[6], (data[3]), label=data[1] + '-' + data[0])
    ax.set(title='Кадр/cек')
    ax.legend(loc='center', fontsize='xx-small', bbox_to_anchor=(0.5, -0.10), shadow=False, ncol=5)
plt.savefig('Кадр_cек.png', dpi=1000)

fig, ax = plt.subplots()
for data in big_data:
    ax.plot(data[6], (data[2]), label=data[1] + '-' + data[0])
    ax.set(title='Загрузка ЦП')
    ax.legend(loc='center', fontsize='xx-small', bbox_to_anchor=(0.5, -0.10), shadow=False, ncol=5)
plt.savefig('ЦП.png', dpi=1000)

fig, ax = plt.subplots()
for data in big_data:
    if data[1] == 'gpu':
        ax.plot(data[6], (data[4]), label=data[1] + '-' + data[0])
        ax.set(title='Загрузка Графического процессора')
        ax.legend(loc='center', fontsize='xx-small', bbox_to_anchor=(0.5, -0.10), shadow=False, ncol=5)
plt.savefig('ГП.png', dpi=1000)

fig, ax = plt.subplots()
for data in big_data:
    ax.plot(data[6], (data[5]), label=data[1] + '-' + data[0])
    ax.set(title='Загрузка ОЗУ в мегабайтах')
    ax.legend(loc='center', fontsize='xx-small', bbox_to_anchor=(0.5, -0.10), shadow=False, ncol=5)
plt.savefig('ОЗУ.png', dpi=1000)