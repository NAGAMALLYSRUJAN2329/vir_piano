from playsound import playsound
import asyncio
# async def fn():
# print('This is ')
# await asyncio.sleep(1)
# print('asynchronous programming')
# await asyncio.sleep(1)
# print('and not multi-threading')
# asyncio.run(fn())
import time
async def play():
    playsound('notes/A0.wav', block=True)
    # await asyncio.sleep(1)
    playsound('notes/A0.wav', block=True)

    print("Audio playback finished.")
    # time.sleep(4)
asyncio.run(play())


# import pyglet
# import time
# sound = pyglet.media.load("notes\A0.wav", "notes\A0.wav")
# sound.play()
# time.sleep(sound.duration)


# import pygame
# pygame.mixer.init()
# pygame.mixer.music.load("notes\A0.wav")
# pygame.mixer.music.play()
# # while pygame.mixer.music.get_busy():
# #     pass
# input("Press Enter to stop playback...")
# pygame.mixer.music.stop()


# import pyaudio
# import wave
# #define stream chunk
# chunk = 1024
# #open a wav format music
# f = wave.open("notes/A0.wav","rb")
# #instantiate PyAudio
# p = pyaudio.PyAudio()
# #open stream
# stream = p.open(format = p.get_format_from_width(f.getsampwidth()),
#                 channels = f.getnchannels(),
#                 rate = f.getframerate(),
#                 output = True)
# #read data
# data = f.readframes(chunk)
# #play stream
# while data:
#     stream.write(data)
#     data = f.readframes(chunk)
# #stop stream
# stream.stop_stream()
# stream.close()
# #close PyAudio
# p.terminate()


# import librosa
# output,fs=librosa.load('notes/A0.wav')

# from scipy.io import wavfile
# fs,output = wavfile.read('notes/A0.wav')

# import sounddevice as sd
# sd.default.samplerate = fs
# sd.play(output)
# # sd.stop()
# sd.wait()
# print("playing")