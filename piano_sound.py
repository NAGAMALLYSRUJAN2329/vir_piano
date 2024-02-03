import pygame

def play_piano_sound(notes):
    file_paths=[f"cut_notes\{note}.wav" for note in notes]
    pygame.display.set_caption('')
    pygame.mixer.init()
    pygame.mixer.stop()
    channels = [pygame.mixer.Channel(i) for i in range(len(file_paths))]
    # pygame.mixer.music.load("notes\A0.wav")
    # pygame.mixer.music.play()
    for i, file_path in enumerate(file_paths):
        sound_effect = pygame.mixer.Sound(file_path)
        channels[i].play(sound_effect)