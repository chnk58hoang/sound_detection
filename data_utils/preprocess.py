from pydub import AudioSegment
from pydub.effects import normalize
from tqdm import tqdm
from typing import List
import os
import scaper
import pandas as pd

def normalize_audio(audio_path: str):
    audio = AudioSegment.from_file(audio_path)
    normalized_audio = normalize(audio)
    normalized_audio = normalized_audio.set_sample_width(2)
    normalized_audio.export(audio_path, format="wav")

def resample_audio(audio_path: str,
                   sample_rate: int = 16000):
    audio = AudioSegment.from_file(audio_path)
    resampled_audio = audio.set_frame_rate(sample_rate)
    resampled_audio.export(audio_path, format="wav")


def concat_files(file_list: list,
                 prefix_str: str,
                 file_dir: str,
                 save_dir: str):
    """Concat audio files 

    Args:
        background_file_list (str): _des
    """
    start = 0
    end = 0
    while start < len(file_list):
        end = start + 9
        if end < len(file_list):
            background_files = file_list[start:end]
            background_files =  [os.path.join(file_dir, file) for file in background_files]
            background_audio = AudioSegment.from_file(background_files[0])
            for background_file in tqdm(background_files[1:],
                                        total=len(background_files) - 1):
                background_audio += AudioSegment.from_file(background_file)
            new_filename = f"{prefix_str}_{start}_{end}.wav"
            new_filepath = os.path.join(save_dir, new_filename)
            background_audio = background_audio.set_frame_rate(16000)
            background_audio.export(new_filepath, format="wav")
            start = end + 1
        else:
            break
        
def split_file_by_prefix(all_filenames: List[str]):
    prefix_file_list = dict()
    for filename in all_filenames:
        prefix = filename[:5]  # 5 first characters
        if prefix not in prefix_file_list:
            prefix_file_list[prefix] = [filename]
        else:
            prefix_file_list[prefix].append(filename)
    return prefix_file_list


def create_mixture_soundscape(machine_sound_path: str,
                              background_sound_path: str,
                              save_path: str,
                              sc: scaper.Scaper):
    
    # normalize_audio(audio_path=background_sound_path)
    sc.add_background(label=('const', 'env'),
                      source_file=('const', background_sound_path),
                      source_time=('const', 0))
    # normalize_audio(audio_path=machine_sound_path)
    sc.add_event(label=('const', 'machine'),
                 source_file=('const', machine_sound_path),
                 source_time=('uniform', 1 ,5),   # start time of the event audio
                 event_time=('uniform', 1, 4),  # start time of the event in the soundscape
                 event_duration=('const', 9),
                 snr=('normal', 8, 3),
                 pitch_shift=None,
                 time_stretch=None)

    sc.generate(audio_path=save_path,
                allow_repeated_label=True,
                allow_repeated_source=True,
                reverb=0.1,
                disable_sox_warnings=True,
                no_audio=False)
    resample_audio(audio_path=save_path)
    normalize_audio(audio_path=save_path)
    
def create_mixture_dataset(source_dir: str,
                           background_dir: str,
                           mixture_dir: str,
                           csv_path: str,
                           split_name):
    df = pd.DataFrame()
    all_source_files = sorted(os.listdir(source_dir))[:100]
    all_background_files = sorted(os.listdir(background_dir))[:100]
    
    if split_name == 'train':
        all_source_files = all_source_files[:int(len(all_source_files) * 0.7)]
        all_background_files = all_background_files[:int(len(all_background_files) * 0.7)]
    elif split_name == 'valid':
        all_source_files = all_source_files[int(len(all_source_files) * 0.7):int(len(all_source_files) * 0.8)]
        all_background_files = all_background_files[int(len(all_background_files) * 0.7):int(len(all_background_files) * 0.8)]
    else:
        all_source_files = all_source_files[int(len(all_source_files) * 0.8):]
        all_background_files = all_background_files[int(len(all_background_files) * 0.8):]
    machine = []
    bg = []
    mix = []
    for idx1, source_file in tqdm(enumerate(all_source_files), total=len(all_source_files)):
        for idx2, background_file in enumerate(all_background_files):
            sc = scaper.Scaper(fg_path=os.path.dirname(source_dir),
                       bg_path=os.path.dirname(background_dir),
                       duration=9,
                       random_state=123)
            sc.ref_db = -20
            mixture_file = f'mixfile_{idx1}_{idx2}.wav'
            mixture_filepath = os.path.join(mixture_dir, mixture_file)
            machine_soundpath = os.path.join(source_dir, source_file)
            bg_soundpath = os.path.join(background_dir, background_file)
            machine.append(source_file)
            bg.append(background_file)
            mix.append(mixture_file)
            create_mixture_soundscape(machine_sound_path=machine_soundpath,
                                      background_sound_path=bg_soundpath,
                                      save_path=mixture_filepath,
                                      sc=sc)
    df['mix'] = mix
    df['machine'] = machine
    df['bg'] = bg
    df.to_csv(csv_path, sep='\t')
     


if __name__ == '__main__':
    BACKGROUND_DIR = "concated_bg/env"
    FOREGROUND_DIR = "concated_fg/machine"
    MIXTURE_DIR = "mixture_test"
    
    create_mixture_dataset(source_dir=FOREGROUND_DIR,
                           background_dir=BACKGROUND_DIR,
                           mixture_dir=MIXTURE_DIR,
                           split_name='test',
                           csv_path='test.csv')
        