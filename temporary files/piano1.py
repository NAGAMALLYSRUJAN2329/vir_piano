import pygame
import time

def play_sound(file_path, volume=1.0):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.set_volume(volume)
    pygame.mixer.music.play()

def play_multiple_sounds(file_paths, volumes=None):
    pygame.mixer.init()
    pygame.init()

    if volumes is None:
        volumes = [1.0] * len(file_paths)

    channels = []
    for i, file_path in enumerate(file_paths):
        channel = pygame.mixer.Channel(i)
        channels.append(channel)
        sound = pygame.mixer.Sound(file_path)
        channel.set_volume(volumes[i])
        channel.play(sound)

    # Wait for sounds to finish
    while any(channel.get_busy() for channel in channels):
        time.sleep(0.1)

if __name__ == "__main__":
    sound_files = ["./notes/Ab4.wav"]
    volumes = [ 1.0]  # Adjust volumes as needed

    # Play sounds simultaneously
    play_multiple_sounds(sound_files, volumes)

    # Or play sounds one by one
    # for sound_file, volume in zip(sound_files, volumes):
    #     play_sound(sound_file, volume)
