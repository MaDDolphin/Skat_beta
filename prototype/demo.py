"""Скрипт для загрузки видео"""
import os
import pytube


def main():
    # -*- coding: utf-8 -*-
    print('https://youtu.be/X3u995Q3-Uw')

    yt = pytube.YouTube('https://youtu.be/X3u995Q3-Uw')
    stream = yt.streams.first()

    new_name = stream.download('./')
    template = "./{number}.mp4"
    os.rename(new_name, template.format(number='demo'))



if __name__ == "__main__":
    main()
