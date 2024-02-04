from pydub import AudioSegment
import os

def cut_wav(input_file, output_file, start_ms, end_ms):
    sound = AudioSegment.from_wav(input_file)
    cut_sound = sound[start_ms:end_ms]
    cut_sound.export(output_file, format="wav")
def get_wav_files(directory):
    wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    return wav_files

if __name__ == "__main__":
    directory_path = "./notes"  # Replace with your directory path

    wav_files = get_wav_files(directory_path)
    for file in wav_files:

        input_file =directory_path+"/"+file  # Replace with your input file path
        output_file = "./cut_notes/"+file  # Replace with your desired output file path
        start_ms = 0  # Replace with the start time in milliseconds
        end_ms = 1000 # Replace with the end time in milliseconds

        cut_wav(input_file, output_file, start_ms, end_ms)

    






# if __name__ == "__main__":
#     directory="./notes"
#     files  in os.listdir(directory):


        # input_file = "input.wav"  # Replace with your input file path
        # output_file = "output.wav"  # Replace with your desired output file path
        # start_ms = 5000  # Replace with the start time in milliseconds
        # end_ms = 10000  # Replace with the end time in milliseconds

        # cut_wav(input_file, output_file, start_ms, end_ms)
