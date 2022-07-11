"""Скрипт для загрузки видео из видеохостинга YouTube для собственного датасета"""
import os
import pytube


def main():
    # -*- coding: utf-8 -*-
    f = open('fight_links.txt')
    i = 1
    for link in f.readlines():
        print(link)
        print(i)
        yt = pytube.YouTube(link)
        stream = yt.streams.first()

        new_name = stream.download('./Fight/')
        template = "./Fight/{number}.mp4"
        os.rename(new_name, template.format(number=i))
        i = i + 1

    b = open('NO_fight_links.txt')
    i = 251
    for link in b.readlines():
        print(link)
        print(i)
        yt = pytube.YouTube(link)
        stream = yt.streams.first()

        new_name = stream.download('./NO_Fight/')
        template = "./NO_fight/{number}.mp4"
        os.rename(new_name, template.format(number=i))
        i = i + 1


if __name__ == "__main__":
    main()
