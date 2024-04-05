import sys
sys.path.insert(0, 'libs')
import libsinstall
libsinstall.install_libraries()

import os
import soundfile as sf
import numpy as np
import time
import random
import logging
import configparser
import json

'''
This code import a logger that saves the logging.log inside the specified
'''
import logger
temp_folder = "__pycache__"
logger.logger(temp_folder)

'''

Should work on both Python 3.x (tested on Python 3.11.6)

This collection is based upon the following packages (auto-installation):
  - numpy
  - pysoundfile
  - random

------------------------
'''

__version__ = "0.01"
__author__  = "G.Salada"

'''
------------------------
'''
# import cfg file
config = configparser.ConfigParser()
config.read('PYgenerator.cfg')


# global
random_q_order = config.getboolean('global', 'random_q_order', fallback=True)
n_questions = config.getint('global', 'n_questions', fallback=0)
n_answers = config.getint('global', 'n_answers', fallback=0)
n_participants = config.getint('global', 'n_participants', fallback=0)
prob_question = config.getfloat('global', 'prob_question', fallback=0.5)
prob_prompt = config.getfloat('global', 'prob_prompt', fallback=0.5)
prob_p_q = config.getfloat('global', 'prob_p_q', fallback=1)
volume = config.get('global', 'volume', fallback="ND")
first_question = config.getboolean('global', 'first_question', fallback=True)
# sex
gender_fixed_quantity = config.getboolean('sex', 'fixed_quantity', fallback=False)
limit_male_female = config.get('sex', 'male_female_ratio', fallback="0:0")
# files
name_format = (config.get('files', 'name_format')).split("_")
dir_path = config.get('files', 'dir_path', fallback=os.path.dirname(os.path.realpath(__file__)))
input_folder = config.get('files', 'input_folder', fallback="INPUT")
output_folder = config.get('files', 'output_folder', fallback="OUTPUT")
import_name1 = os.path.join("custom", config.get('files', 'custom_path', fallback="output_files.json"))
noise_file = config.get('files', 'noise_file', fallback="")
import_name_s = os.path.join("custom", config.get('files', 'custom_burst', fallback="burst.json"))
# fade
fade_length = config.getfloat('fade', 'fade_length', fallback=0)
fade_type = config.getint('fade', 'fade_type', fallback=0)
# pauses
p_min = config.getfloat('pauses', 'min', fallback=0.5)
p_max = config.getfloat('pauses', 'max', fallback=1.0)
# first pause
lp_min = config.getfloat('first pause', 'min', fallback=p_min)
lp_max = config.getfloat('first pause', 'max', fallback=p_max)
# silences
s_min = config.getfloat('silences', 'min', fallback=0.05)
s_max = config.getfloat('silences', 'max', fallback=0.10)
# long silences
ls_min = config.getfloat('long silences', 'min', fallback=1.0)
ls_max = config.getfloat('long silences', 'max', fallback=2.0)
# burst
s_quantity = config.getfloat('burst', 'b_quantity', fallback=random.random())
min_s_distance = config.getfloat('burst', 'min_b_distance', fallback=4)
cut_redundancy = config.getfloat('burst', 'cut_redundancy', fallback=1.5)
sound_amp_fact = config.getfloat('burst', 'sound_amp_fact', fallback=1)
cycle_limit = config.getint('burst', 'cycle_limit', fallback=10)
closest_distance = config.getfloat('burst', 'similar_distance', fallback=2)
on_conjunction = config.getfloat('burst', 'near_conjunction', fallback=1)
# data
sample_rate = config.getint('data', 'sample_rate', fallback=0)
channels = config.getint('data', 'sample_rate', fallback=0)
pop_tollerance = sample_rate * 1
save_name1 = os.path.join(temp_folder, "output_files.json")
save_name_s = os.path.join(temp_folder, "burst.json")
pos_participants = {}

''' some checks before the program starts '''
if not config.has_option('files', 'custom_path'): custom_files_enabler = False
else: custom_files_enabler = True
if not config.has_option('files', 'custom_burst'): custom_burst_enabler = False
else: custom_burst_enabler = True
if (volume != "ND" and volume != "H" and volume != "L"):
    raise Exception(f"cfg_check - volume ({volume}) should be ND, H or L")
if gender_fixed_quantity == True and limit_male_female == "0:0":
    raise Exception(f"cfg_check - fixed_quantity should not be {gender_fixed_quantity} if limit_male_female is empty ({limit_male_female})")

'''//////////////////////////////////////////'''


class myrand:
    """
    A custom random number generator with adjustable seed.

    This class provides methods for generating random numbers
    while allowing control over the seed for reproducibility.

    Example usage:
    ```
    mr = myrand()
    print(mr.randint(1, 100))
    ```
    """

    def __init__(self):
        self.global_seed_counter = int(time.time() * 7) % 100
        self.randint_consecutive_count = 0
        self.randint_last_generated = None
    def seed_changer(self):
        """This code is completely random, just change the seed to something"""
        if self.global_seed_counter>50:
            self.global_seed_counter += 37
        else:
            self.global_seed_counter += 27
        if self.global_seed_counter > 99:
            self.global_seed_counter %= 100
    def choice(self, a):
        """choice: Choose a random element from a non-empty sequence."""
        self.seed_changer()
        random.seed(self.global_seed_counter)
        return random.choice(a)
    def uniform(self, a, b) -> float:
        """uniform: Get a random number in the range [a, b) or [a, b] 
        depending on rounding."""
        self.seed_changer()
        random.seed(self.global_seed_counter)
        return random.uniform(a, b)
    def randint(self, a, b, p1=1):
        """
        Return a random integer N such that a <= N <= b.

        :param a: Lower bound.
        :param b: Upper bound.
        :param p1: Probability on the first lower number
        :return: Random integer.
        """
        self.seed_changer()
        random.seed(self.global_seed_counter)
        # probability for 1 is lower
        output = random.randint(a, b)
        if output == self.randint_last_generated:
            self.randint_consecutive_count += 1
            if self.randint_consecutive_count >= 1:
                self.seed_changer()
                random.seed(self.global_seed_counter)
                output = random.randint(a, b)
        else:
            self.randint_last_generated = output
            self.randint_consecutive_count = 0
        if output == a and random.random() < p1:
            return a
        else:
            return random.randint(a+1, b)
    def random(self):
        """random: x in the interval [0, 1)."""
        self.seed_changer()
        random.seed(self.global_seed_counter)
        return random.random()
    
myrand = myrand()

def check_limits():
    try:
        limit = limit_male_female.split(":")
        if len(limit) != 2:
            raise Exception("check_limits - limit_male_female should be 'number1:number2'.")
        limit_male, limit_female = map(int, limit)
        if limit_male < 0 or limit_female < 0:
            raise Exception("check_limits - limit_male_female numbers must be positives")
        return limit_male, limit_female
    except ValueError:
        raise Exception(f"check_limits - limit_male_female should be 'number1:number2'.")

def audio_file(path: str, length: float, name: str, person: str, duplicated: bool):
    file = {}
    file['path'] = path
    file['length'] = length
    file['name'] = name
    file['person'] = person
    file['duplicated'] = duplicated
    logging.info(f"audio_file \t\t - SUCCESS for: {path}: {type(file)}")
    return file

def get_person(filename: str):
    filename_without_extension = os.path.splitext(os.path.basename(filename))[0]
    logging.info(f"get_person \t\t - SUCCESS for: {filename}")
    return filename_without_extension.split("_")[(name_format.index('person'))]

def get_gender(filename: str):
    filename_without_extension = os.path.splitext(os.path.basename(filename))[0]
    logging.info(f"get_gender \t\t - SUCCESS for: {filename}")
    return filename_without_extension.split("_")[name_format.index('gender')]

def get_type(filename: str):
    filename_without_extension = os.path.splitext(os.path.basename(filename))[0]
    logging.info(f"get_type \t\t - SUCCESS for: {filename}")
    return filename_without_extension.split("_")[name_format.index('type')]

def get_nquestion(filename: str):
    filename_without_extension = os.path.splitext(os.path.basename(filename))[0]
    logging.info(f"get_type \t\t - SUCCESS for: {filename}")
    return filename_without_extension.split("_")[name_format.index('question')]

def get_volume(filename: str):
    filename_without_extension = os.path.splitext(os.path.basename(filename))[0]
    logging.info(f"get_volume \t\t - SUCCESS for: {filename}")
    return filename_without_extension.split("_")[name_format.index('volume')]

def get_channels(data):
    '''Get the number of channels in an audio file'''
    if len(np.shape(data)) == 1:
        logging.info(f"get_channels \t\t - SUCCESS: {1}")
        return 1
    else:
        logging.info(f"get_channels \t\t - SUCCESS: {np.shape(data)[1]}")
        return np.shape(data)[1]

def add_file(file_names, file):
    '''add file to file_names array and use audio_file() function'''
    person = get_person(file)
    duplicated = False
    if person in [file_names[i]['person'] for i in range(len(file_names))]:
        duplicated = True
    file_names.append(audio_file(file, 0, os.path.splitext(os.path.basename(file))[0], person, duplicated))
    logging.info(f"add_file \t\t - SUCCESS for: {file}")
    return file_names

def find_file(name, path):
    '''Search for the file by its name with and without the extension
    and return the first file found with the exact path of the file.'''
    for root, _, files in os.walk(path):
        for file in files:
            if name in file and (file.lower().endswith(".wav")):
                logging.info(f"find_file \t\t - SUCCESS.")
                return os.path.join(root, file)
    raise Exception(f"File {name}.wav not found in {path}")

def folder_info(folder_path):
    '''count the number of audio files in a folder and split questions 
    from answers, burst and prompts '''
    max_files = 0
    count_q = [] #array with all questions
    count_a = [] #array with all answers
    count_iq = [] #array with all initial answer
    count_s = [] #array with all burst
    q_letters = {} #count questions persons
    a_letters = {} #count answers persons
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.wav'):
                if len(filename.split('_')) > len(name_format):
                    logging.info(f"folder_info \t\t - WARNING: {filename} name format is NOT correct. See manual for correct file naming.\n")
                else: 
                    file_type = get_type(filename)
                    if file_type == "B":
                        count_s.append(filename)
                    elif volume == "ND" or get_volume(filename) == volume:
                        if file_type == "Q":
                            count_q.append(filename)
                            person = get_person(filename)
                            q_letters[person] = get_nquestion(filename)
                        elif file_type == "A":
                            count_a.append(filename)
                            person = get_person(filename)
                            a_letters[person] = get_nquestion(filename)
                        elif file_type == "P":
                            count_iq.append(filename)
                    max_files += 1
    if count_q == []:
        logging.info(f"folder_info \t\t - No Initial Questions.")
        raise Exception(f"\n No questions in the folder.")
    if count_a == []:
        logging.info(f"folder_info \t\t - No Initial Questions.")
        raise Exception(f"\n No answers in the folder.")
    if count_iq == []:
        logging.info(f"folder_info \t\t - No Initial Questions.")
        raise Exception(f"\n No Initial Questions in the folder.")
    if len(q_letters) == 1 and len(a_letters) == 1 and q_letters.keys() == a_letters.keys():
        logging.info(f"folder_info \t\t - Only one participant.")
        raise Exception(f"\n Only one participant!!!")
    logging.info(f"folder_info \t\t - SUCCESS: max_files: {max_files}, \tcount_a: {count_a}, \tcount_q: {count_q}, \tcount_iq: {count_iq}, \tcount_s: {count_s}, \ta_letters: {a_letters}, \tq_letters: {q_letters}")
    if n_participants != 0:
        a_letters, q_letters = participants_handler(a_letters, q_letters)
        count_a = filter_participants(count_a, a_letters)
        count_q = filter_participants(count_q, q_letters)
        count_iq = filter_participants(count_iq, q_letters)
    return max_files, count_a, count_q, count_iq, count_s, a_letters, q_letters

def participants_handler(a_letters, q_letters):
    '''if n_participants was defined, choose 2 participants'''
    a_participants = {}
    q_participants = {}
    a_list = list(a_letters.keys())
    q_list = list(q_letters.keys())
    array = []
    if n_participants == 1:
        raise Exception("n_participants too low")
    elif n_participants == 2:
        tmp_cycle_limit = 0
        key_a = random.choice(a_list)
        a_participants[key_a] = a_letters.pop(key_a)
        array.append(key_a)
        while True:
            key = random.choice(q_list)
            if key_a != key:
                break
            elif tmp_cycle_limit > 10:
                raise Exception('error with n_participants')
            tmp_cycle_limit += 1
        array.append(key)
        q_participants[key] = q_letters.pop(key)
    else:
        array = []
        for i in range(n_participants):
            if i % 2 == 0:
                key = random.choice(a_list)
                a_list.remove(key)
                if key not in array:
                    array.append(key)
                    if len(array) > n_participants:
                        break
                a_participants[key] = a_letters.pop(key)
            else:
                key = random.choice(q_list)
                q_list.remove(key)
                if key not in array:
                    array.append(key)
                    if len(array) > n_participants:
                        break
                q_participants[key] = q_letters.pop(key)
    for i in array:
        if i in a_letters.keys():
            a_participants[i] = a_letters.pop(i)
        if i in q_letters.keys():
            q_participants[i] = q_letters.pop(i)
    return a_participants, q_participants

def filter_participants(tmp_count, tmp_participants):
    filtered_count = [filename for filename in tmp_count if get_person(filename) in tmp_participants]
    return filtered_count

def burst_info(folder_path):
    '''count the number of sound files in a folder and split 
    questions from answers.\n
    It is a better optimized folder_info version for burst files.'''
    count_s = []
    for _, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.wav'):
                if len(filename.split('_')) > len(name_format):
                    logging.info(f"burst_info \t\t - WARNING: {filename} name format is NOT correct. See manual for correct file naming.\n")
                else: 
                    file_type = get_type(filename)
                    if file_type == "B":
                        count_s.append(filename)
    return count_s

def check_SR_CH(name, rate_temp, channels_temp):
    '''handle SampleRate and Channels Exceptions
    to avoid unuseful file reads,
    this function handle only Exceptions'''
    if sample_rate != rate_temp:
        raise Exception(f"\n Audio file n{name} has different sample rate (must be {rate_temp} hz).")
    elif channels != channels_temp:
        raise Exception(f"\n Audio file n{name} has different channels number (must be {channels_temp} ch).")
    else:
        logging.info(f"check_SR_CH \t\t - SUCCESS for {name}")

def ceil(value) -> int:
    if isinstance(value, int):
        integer = int(value)
    elif isinstance(value, float):
        integer, decimal = str(value).split(".")
        if int(decimal) != "0":
            integer = int(integer) + 1
    return integer

def raw_to_seconds(audio):
    '''audio data to length. Works with STEREO and MONO'''
    duration = len(audio) / sample_rate
    logging.info(f"raw_to_seconds \t\t - SUCCESS")
    return duration

def shape_fixer(mono_array, shape):
    if shape > 1:
        stereo_array = np.empty((mono_array.shape[0], channels), dtype=mono_array.dtype)
        for i in range(channels):
            stereo_array[:, i] = mono_array
    else:
        stereo_array = mono_array.reshape(-1)
    return stereo_array

def noise(raw_file):
    '''just adds a background user chosen noise'''
    raw_noise, sample_rate = sf.read(dir_path + "/" + noise_file)
    temp_channels = get_channels(raw_noise)
    check_SR_CH(noise_file, sample_rate, temp_channels)
    raw_length = len(raw_file)
    noise_length = len(raw_noise)
    if noise_length < raw_length:
        n_repeats = int(raw_length/noise_length)+1
        tmp_noise = raw_noise
        for _ in range(n_repeats):
            raw_noise = concatenate_fade(raw_noise, tmp_noise, len(raw_noise.shape))
    total = np.add(raw_noise[:raw_length], raw_file)
    if len(raw_noise[:raw_length]) != len(raw_file) or len(raw_file) != len(total):
        raise Exception("INTERNAL ERROR: function 'noise', length does not match")
    logging.info(f"noise \t\t\t - SUCCESS.")
    return total

def concatenate_fade(audio1, audio2, shape):
    '''Concatenate two audio files using a noise as a bandade (two fade-in, fade-out)'''
    fade = int(fade_length * sample_rate)
    if noise_file != "":
        noise = sf.read(dir_path + "/" + noise_file)[0]
        if len(noise) < (fade * 2):
            fade = int(len(noise) / 2)
            logging.info(f"concatenate_fade \t\t - NOTE: fade = {fade}")
    if len(audio1) == 0:
        logging.info(f"concatenate_fade \t\t - WARNING: audio1 is empty")
        return audio2
    elif len(audio2) == 0:
        logging.info(f"concatenate_fade \t\t - WARNING: audio2 is empty")
        return audio1
    elif len(audio1) < (fade):
        logging.info(f"concatenate_fade \t\t - WARNING: audio1 ({len(audio1)} samples) smaller than fade ({fade} samples)!! fade will set to 0.05!!!")
        fade = int(0.05 * sample_rate)
    elif len(audio2) < (fade):
        logging.info(f"concatenate_fade \t\t - WARNING: audio2 ({len(audio1)} samples) smaller than fade ({fade} samples)!! fade will set to 0.05!!!")
        fade = int(0.05 * sample_rate)
    # applica il fade-out al primo file audio
    if fade_type == 0:
        fade_out = np.logspace(np.log10(0.15), np.log10(1.05), fade)
    elif fade_type == 1:
        fade_out = np.linspace(np.log10(0.15), np.log10(1.05), fade)
    elif fade_type == 2:
        fade_out = np.geomspace(np.log10(0.15), np.log10(1.05), fade)
    #fade_out = np.linspace(0.15, 1.05, fade)
    fade_out = shape_fixer(np.subtract(1.1, fade_out), shape)
    fade_in = np.flip(fade_out)
    FO_audio1 = audio1[-fade:] * fade_out
    FI_audio2 = audio2[:fade] *fade_in
    if noise_file != "":
        noise = noise[:(fade*2)]
        FI_rumore = noise[:fade]*fade_in
        FO_rumore = noise[-fade:]*fade_out
        # Unisci i file audio
        mixed_audio1 = np.add(FO_audio1, FI_rumore)
        mixed_audio2 = np.add(FI_audio2, FO_rumore)
        audio1[-fade:] = mixed_audio1
        audio2[:fade] = mixed_audio2
    else:
        audio1[-fade:] = audio1[-fade:] * fade_out
        audio2[:fade] = audio2[:fade] * fade_in
    #concatena i files audio
    if channels > 1:
        OUTPUT = np.concatenate((audio1, audio2), axis=0)
    else:
        OUTPUT = np.concatenate((audio1, audio2))
    if len(audio1) + len(audio2) != len(OUTPUT):
        raise Exception("INTERNAL ERROR: concatenate_fade function, length does not match")
    logging.info(f"concatenate_fade \t - SUCCESS")
    return OUTPUT

def concatenate(data1, data2, pause_length):
    '''join 2 audio files data1, data2 and adds a pause between them
    handles noises or fade-in/fade-out'''
    n_sample_silence = int(sample_rate * pause_length)
    shape = len(data1.shape)

    if shape > 1: #stereo
        silence = np.zeros((n_sample_silence, channels))
    else: #mono
        silence = np.zeros((n_sample_silence,))
    if noise_file != "":
        silence = noise(silence)
        OUTPUT = concatenate_fade(data1, silence, shape)
        OUTPUT = concatenate_fade(OUTPUT, data2, shape)
    else:
        if fade_length == 0:
            if shape > 1: #stereo
                OUTPUT = np.concatenate((data1, silence, data2), axis=0)
            else: #mono
                OUTPUT = np.concatenate((data1, silence, data2))
        else:
            OUTPUT = concatenate_fade(data1, silence, shape)
            OUTPUT = concatenate_fade(OUTPUT, data2, shape)
    logging.info(f"concatenate \t\t - SUCCESS")
    return OUTPUT

def silence_generator(file_names):
    '''Generate long pause, short pause or silence in seconds'''
    print("\t adding pauses and silences", end="")
    silences = []
    tmp_count_questions = 0
    previous_question = None
    first_file = file_names[0]['name']
    first_type = get_type(first_file)
    
    point_counter = 0
    point_length = len(file_names)
    for i in range(len(file_names) - 1):
        point_counter += 1
        if point_counter % (point_length // 3) == 0: print(".", end="")
        if i!= 0:
            first_file = second_file
            first_type = second_type
        second_file = file_names[i+1]['name']
        second_type = get_type(second_file)
        # count how many questions there are of the same question
        actual_question = get_nquestion(first_file)
        if actual_question != previous_question:
            previous_question = actual_question
            tmp_count_questions = 0
        else: tmp_count_questions += 1

        if second_type == "A":
            if tmp_count_questions == 0:
                pause_length = myrand.uniform(lp_min, lp_max)  # seconds
            else:
                pause_length = myrand.uniform(p_min, p_max)  # seconds
        elif first_type == "I" and second_type == "Q":
                # if the pause is a SILENCE
                pause_length = myrand.uniform(s_min, s_max)
        elif first_type == "A":
            # if the pause is after an answer (LONG PAUSE)
            pause_length = myrand.uniform(ls_min, ls_max)  # seconds
        silences.append(pause_length)
    logging.info(f"silence_generator \t - SUCCESS.")
    print()
    return silences

def file_complete(file_names, silences, audio_data):
    '''add pause and join elements full or empty base on i value'''
    for j in range(len(file_names)):
        tmp_name = file_names[0]['name']
        if j == 0:
            OUTPUT = audio_data[tmp_name]
        else:
            OUTPUT = concatenate(OUTPUT, audio_data[tmp_name], silences[j - 1])
    logging.info(f"file_complete \t\t - SUCCESS.")
    return OUTPUT

def file_names_reader(file_names):
    print("\t reading audio data", end="")
    global channels, sample_rate
    file_data_dict = {}
    point_counter = 0
    point_length = len(file_names)
    for item in file_names:
        point_counter += 1
        if point_counter % (point_length // 3) == 0: print(".", end="")
        file_path = item['path']
        file_name = item['name']
        temp_data, temp_rate = sf.read(file_path)
        if channels == 0 or sample_rate == 0:
            channels += get_channels(temp_data)
            sample_rate += temp_rate
        else:
            check_SR_CH(file_name, temp_rate, get_channels(temp_data))
        file_data_dict[file_name] = temp_data
    logging.info(f"file_names_reader \t - SUCCESS.")
    return file_data_dict

def dialogs_join(file_names:list, silences:list):
    ''' MAIN FUNCTION: create the ending file;
    create the class inside file_names and return to concatenate(). 
    check channels, sample_rate'''

    # Add audio data and check sample rate and channels
    audio_data = file_names_reader(file_names)
    for i in range(len(file_names)):
        file_names[i]['length'] = round(len(audio_data[file_names[i]['name']]))
    print(" creating data", end="")
    OUTPUT2 = []
    # ALTERNATIVE SUM GENERATOR
    if noise_file != "":
        OUTPUT = file_complete(file_names, silences, audio_data)
        OUTPUT2.append([OUTPUT, "COMPLETE"])
    OUTPUT = set()
    len_file_names = len(file_names)
    # i è l'elemento da stampare con i dati, mentre j è l'elemento attuale
    point_counter = 0
    point_length = len_file_names
    for i in range(len_file_names):
        point_counter += 1
        if point_counter % (point_length // 3) == 0: print(".", end="")
        print_person = file_names[i]['person']
        if not file_names[i]['duplicated']:
            print_name = print_person
            for j in range(len_file_names):
                tmp_name = file_names[j]['name']
                tmp_person = file_names[j]['person']
                tmp_length2 = file_names[j]['length']
                if j == 0:
                    # gestisce il primo elemento e lo mette vuoto se non è lui
                    if tmp_person == print_person:
                        OUTPUT = audio_data[tmp_name]
                        del audio_data[tmp_name]
                    else:
                        if channels > 1:
                            OUTPUT = np.zeros((tmp_length2, channels))
                        else:
                            OUTPUT = np.zeros((tmp_length2,))
                        if noise_file != "":
                            OUTPUT = noise(OUTPUT)
                else:
                    # aggiunge pausa e concatena elementi pieni o vuoti in base al valore di i.
                    if tmp_person == print_person:
                        OUTPUT = concatenate(OUTPUT, audio_data[tmp_name], silences[j - 1])
                        del audio_data[tmp_name]
                    else:
                        if channels > 1:
                            file_silence = np.zeros((tmp_length2, channels))
                        else:
                            file_silence = np.zeros((tmp_length2,))
                        if noise_file != "":
                            file_silence = noise(file_silence)
                        OUTPUT = concatenate(OUTPUT, file_silence, silences[j - 1])
            OUTPUT2.append([OUTPUT, print_name])
            logging.info(f"dialogs_join \t - SUCCESS for: {print_name}")
    for h in OUTPUT2:
        if h[1] != "COMPLETE":
            if 'total_data' not in locals():
                total_data = h[0]
            else:
                total_data = np.add(total_data, h[0])
    if noise_file == "":
        OUTPUT2.append([total_data, "COMPLETE"])
    print()
    logging.info(f"filenames_lengths \t - SUCCESS")
    #print(sys.getsizeof(OUTPUT) / (1024 * 1024))
    return OUTPUT2


# /////////////////////////////////// burst //////////////////////////////////

def filenames_lengths(file_names, silences):
    '''Create array for each output file with path, person, start and end
    also handle silences, run also if silences is empty'''
    arr = []
    lengh_end, length_start = 0, 0
    i = 0
    for filename in file_names:
        len_file = round(filename['length']) / sample_rate
        lengh_end = len_file + lengh_end
        arr.append([filename["path"], filename["person"], length_start, lengh_end])
        length_start = lengh_end
        if len(silences) != 0:
            if i != (len(file_names)-1):
                length_start = length_start + silences[i]
                lengh_end = lengh_end + silences[i]
                i += 1
    logging.info(f"filenames_lengths \t - SUCCESS: {arr}")
    return arr

def check_length(output_length, max_duration, name, threshold=3):
    limit_length = ceil(max_duration*sample_rate)
    if output_length > limit_length+threshold or output_length < limit_length-threshold:
        logging.info(f"check_length \t\t - ERROR for: {name}")
        raise Exception (f"INTERNAL ERROR: {name} (with length: {output_length}) does not match {limit_length} length")

def handle_s_strangers(sound_files, participants):
    '''remove all persons that does not speak in the dialogue'''
    for filename in sound_files:
        person = get_person(filename)
        if person not in participants:
            sound_files.remove(filename)
    return sound_files

def handle_s_quantity(sound_files):
    '''This code add sound files. The first time it adds all files,
    then proceed shuffling'''
    length_before = len(sound_files)
    int_s_quantity = int(s_quantity)
    tmp_sound_files2 = []
    tmp_sound_files1 = []
    random.shuffle(sound_files)
    for i in range(int_s_quantity+1):
        if i == (int_s_quantity):
            sound_files = sound_files[:int(len(sound_files)*(s_quantity%1))]
        if i == 0 or i == (int_s_quantity):
            tmp_sound_files1 += sound_files
        else:
            # run only on elements that aren't the first or the last element
            for _ in range(len(sound_files)):
                tmp_sound_files2.append(myrand.choice(sound_files))
    random.shuffle(tmp_sound_files2)
    tmp_sound_files1 += tmp_sound_files2
    logging.info(f"handle_s_quantity \t - SUCCESS: {length_before} -> {len(tmp_sound_files1)}")
    return tmp_sound_files1

def closest_calculator(tmp_dict, delay):
    """search which sound is the nearest.
    if a sound is too near another, return None, None touple"""
    closest_key = ""
    closest_value = 24840000
    final_value = 0
    for key, value in tmp_dict.items():
        distance = abs(value-delay)
        if distance < min_s_distance:
            return value, None
        if distance < closest_value:
            closest_value = distance
            final_value = value
            closest_key = key
    return final_value, closest_key

def handle_burst(sound_files, audio_length:list, sound_length:dict, max_duration) -> list:
    '''create 2D list of burst. Each sound has a random position in seconds.\n
    There are various values to setup the randomness.\n
    note: sound_files != sound_length because the first is the final sound number'''
    OUTPUT = []
    tmp_dict = {}
    count_burst = 0
    for filename in sound_files:
        count_burst += 1
        tmp_limit = 0
        person = get_person(filename)
        while True:
            correct = True
            delay = myrand.uniform(0, (max_duration-cut_redundancy-sound_length[filename]))
            if tmp_limit <= cycle_limit:
                # check if a sound is near another, also search which sound is the nearest
                closest_value, closest_key = closest_calculator(tmp_dict, delay)
                if closest_key is None:
                    tmp_limit += 1
                    correct = False
                    logging.info(f"handle_burst \t\t - INFO: tmp_limit: {tmp_limit}: {int(delay//60)}:{int(delay%60)} sound too close to another {int(closest_value//60)}:{int(closest_value%60)}")
                # check if a sound is near another same sound
                if correct != False:
                    if closest_key == filename and abs(closest_value-delay) < closest_distance:
                        correct = False
                        tmp_limit += 1
                        logging.info(f"handle_burst \t\t - INFO: tmp_limit: {tmp_limit}: {int(delay//60)}:{int(delay%60)} same sound near another {int(closest_value//60)}:{int(closest_value%60)}")
                        break
                # check if the sound is inside an empty area of your person
                if correct != False:
                    for i_a in range(len(audio_length)):
                        delay_end = delay + float(sound_length[filename])
                        start_NO_real = audio_length[i_a][2]
                        end_NO_real = audio_length[i_a][3]
                        start_NO_zone = start_NO_real-float(cut_redundancy)
                        end_NO_zone = end_NO_real+float(cut_redundancy)
                        if on_conjunction != 0:
                            if (abs(delay - start_NO_zone) < on_conjunction and abs(delay_end - start_NO_zone) < on_conjunction) or (abs(delay - end_NO_zone) < on_conjunction and abs(delay_end - end_NO_zone) < on_conjunction):
                                correct = False
                                tmp_limit += 1
                                logging.info(f"handle_burst \t\t - INFO: tmp_limit: {tmp_limit}: {int(delay//60)}:{int(delay%60)} sound is too close to a conjunction between burst ({int(audio_length[i_a][3]//60)}:{int(audio_length[i_a][3]%60)})")
                                break
                        if audio_length[i_a][1] == person and delay > (start_NO_zone) < delay and delay_end < (end_NO_zone):
                            correct = False
                            tmp_limit += 1
                            logging.info(f"handle_burst \t\t - INFO: tmp_limit: {tmp_limit}: {int(delay//60)}:{int(delay%60)} sound is inside an audio of the same person")
                            break
            else:
                logging.info(f"handle_burst \t\t - WARNING: no more space for new burst with cycle repeated {tmp_limit}. Added {len(OUTPUT)} burst")
                print(f" added {count_burst} burst...", end ="")
                return OUTPUT
            if correct:
                tmp_dict[filename] = delay
                OUTPUT.append([os.path.join(dir_path, input_folder, filename), person, delay])
                logging.info(f"handle_burst \t\t - NOTE: appended {filename} on position {delay} with cycle repeated {tmp_limit}")
                tmp_limit = 0
                break
    logging.info(f"handle_burst \t\t - SUCCESS: arr expanded with {count_burst} burst.")
    print(f"\t added {count_burst} burst...", end=" ")
    return OUTPUT

def sound_reader(sound_names):
    '''Read audio file and put in array'''
    burst = {}
    length_data = {}
    for file in sound_names:
        path_file = find_file(file, os.path.join(dir_path, input_folder))
        data = sf.read(path_file)[0]
        burst[file] = data
        length_data[file] = raw_to_seconds(data)
    logging.info(f"sound_reader \t - SUCCESS: {burst}, {length_data}")
    return burst, length_data

def burst_concatenate(audio_no_s, burst: list, sound_data:dict, max_duration:float):
    print(f" joining dialogue", end="")
    output = []
    point_counter = 0
    point_length = len(audio_no_s)
    for i in audio_no_s:
        point_counter += 1
        if point_counter % (point_length // 3) == 0: print(".", end="")
        name = i[1]
        if name != "COMPLETE":
            total = i[0]
            for j in burst:
                # if the person is the same
                if name == j[1]:
                    sound = sound_data[os.path.basename(j[0])]
                    if sound_amp_fact != 1.0:
                        sound = sound * sound_amp_fact
                    start_sound = int(float(sample_rate)*j[2])
                    end_sound = len(sound)+start_sound
                    shape = len(total.shape)
                    if noise_file != "":
                        total_tmp = concatenate_fade(total[:start_sound], sound, shape)
                        total = concatenate_fade(total_tmp, total[end_sound:], shape)
                    elif fade_length == 0:
                        if channels > 1: #stereo
                            total = np.concatenate((total[:start_sound-1], sound, total[end_sound-1:]), axis=0)
                        else: #mono
                            total = np.concatenate((total_tmp[:start_sound-1], sound, total[end_sound-1:]))
                    else:
                        total_tmp = concatenate_fade(total[:(start_sound-1)], sound, shape)
                        total = concatenate_fade(total_tmp, total[(end_sound-1):], shape)
            check_length(len(total), max_duration, name)
            output.append([total, name])
    for h in output:
        if h[1] != "COMPLETE":
            if 'total_data' not in locals():
                total_data = h[0]
            else:
                total_data = np.add(total_data, h[0])
    output.append([total_data, "COMPLETE"])
    logging.info(f"burst_concatenate \t - SUCCESS for: {name}")
    print()
    return output

def custom_burst() -> tuple:
    '''if there is a custom burst setting into the folder, 
    proceed to read the settings. \n
    return "list, list"'''
    sound_files = []
    try:
        with open(import_name_s, "r") as file_json:
            data = json.load(file_json)
    except FileNotFoundError:
        raise Exception(f"file {import_name_s} not found.")
    if isinstance(data, dict):
        if all(isinstance(key, str) for key in data.keys()) and all(isinstance(value, float) for value in data.values()):
            output = []
            not_found = 0
            folder = os.path.join(dir_path, input_folder)
            for key, value in data.items():
                try:
                    file = find_file(key, folder)
                    if key not in sound_files:
                        sound_files.append(key)
                except Exception as e:
                    not_found +=1 
                    logging.info(f"custom_burst \t - INFO: {key} not found")
                output.append([file, get_person(key), value])
            if len(output)==0:
                raise Exception('file JSON does not contain a valid dict.')
            if not_found != 0:
                print("Some files are unavaible, open 'logging.log' to see details")
            logging.info(f"custom_burst \t - INFO: found correct dict")
            return output, sound_files
        else:
            raise Exception('file JSON does not contain a valid dict.')
    elif isinstance(data, list):
        for row in data:
            if not isinstance(row, list):
                raise Exception('file JSON does not contain a valid list.')
            file = os.path.basename(row[0])
            if file not in sound_files:
                sound_files.append(file)
        if len(data)==0:
            raise Exception('file JSON does not contain a valid list.')
        logging.info(f"custom_burst \t - INFO: found correct list")
        return data, sound_files
    else: 
        raise Exception('file JSON does not contain a valid dict or list')

def burst(file_names, audio_no_s, silences):
    '''core function called to add burst into the dialogue'''
    if custom_burst_enabler:
        burst, sound_files = custom_burst()
        max_duration = raw_to_seconds(audio_no_s[-1][0])
        sound_data, sound_length = sound_reader(sound_files)
    elif s_quantity == 0:
        logging.info(f"burst \t\t\t - ABORT: s_quantity = 0")
        return audio_no_s
    else:
        sound_files = burst_info(os.path.join(dir_path, input_folder))
        # -1 takes the last file (COMPLETE)
        print("\t adding new burst...", end="")
        tmp_participants = []
        for i in audio_no_s:
            tmp_participants.append(i[1])
        max_duration = raw_to_seconds(audio_no_s[-1][0])
        sound_files = handle_s_strangers(sound_files, tmp_participants)
        sound_data, sound_length = sound_reader(sound_files)
        sound_files = handle_s_quantity(sound_files)
        audio_length = filenames_lengths(file_names, silences)
        burst = handle_burst(sound_files, audio_length, sound_length, max_duration)
        with open((save_name_s), 'w') as file:
            json.dump(burst, file)
    return burst_concatenate(audio_no_s, burst, sound_data, max_duration)

# /////////////////////////////////// burst //////////////////////////////////

def participants_lists(q_letters:dict, a_letters:dict) -> tuple:
    """Returns a tuple of two list of participants"""
    q_participants = list(q_letters.keys())
    a_participants = list(a_letters.keys())
    logging.info(f"participants_lists \t - SUCCESS: {q_participants}, {a_participants}")
    return q_participants, a_participants

def list_to_3Dlist(list1:list):
    """create 3D list for dicts"""
    arr1 = []
    for filename in list1:
        person = get_person(filename)
        n_question = get_nquestion(filename)
        real_path = find_file(filename, os.path.join(dir_path, input_folder))
        arr1.append([real_path, person, n_question])
    logging.info(f"list_to_3Dlist\t\t - SUCCESS")
    return arr1

def questions_shuffler(matr1:list):
    list1 = []
    for _, _, n in matr1:
        if n not in list1:
            list1.append(n)
    if random_q_order:
        random.shuffle(list1)
    if n_questions == 0:
        list1 = list1[:(myrand.randint(1,len(list1), 0.5))]
    elif n_questions < 0:
        length = myrand.randint(1,abs(int(n_questions)), 0.3)
        list1 = list1[:(length)]
    else:
        list1 = list1[:(n_questions)]
    return list1

def matr_to_dict1(matr1:list, list1:list):
    dict1 = {}
    for j in list1:
        for _, p, n in matr1:
            if j == n:
                dict1.setdefault(n, []).append(p)
    return dict1

def volume_handler(p1:str, p2:str):
    """In a "ND" case, if two people are near each other
    , set the volume to the min, else max."""
    if volume == "ND":
        if pos_participants.get(p1) == pos_participants.get(p2):
            tmp_volume = "L"
        else:
            tmp_volume = "H"
        return tmp_volume
    else: return volume

def merge_arrays(arr1:list, arr2:list):
    merged_array = []
    for item in arr1:
        if item not in merged_array:
            merged_array.append(item)
    for item in arr2:
        if item not in merged_array:
            merged_array.append(item)
    logging.info(f"merge_arrays\t\t - SUCCESS: length:{len(merged_array)}")
    return merged_array

def find_gender(participants:list):
    '''Search for the file by its name with and without the extension
    and return the first file found with the exact path of the file.'''
    gen_participants = {}
    path = os.path.join(dir_path, input_folder)
    for _, _, files in os.walk(path):
        for file in files:
            if len(participants) == 0:
                return gen_participants
            if file.lower().endswith(".wav"):
                file_person = get_person(file)
                if file_person in participants:
                    gen_participants[file_person] = get_gender(file)
                    participants.remove(file_person)
                    logging.info(f"find_gender \t\t - SUCCESS.")
    return gen_participants

def handle_M_F(dist_answerers:list, limit_male:int, limit_female:int, tmp_n_answers:int, gen_participants:dict):
    """Handle the limit_male and limit_female"""
    real_limit_male = 0; real_limit_female = 0
    if gender_fixed_quantity != True: 
        if tmp_n_answers == 1:
            logging.info(f"handle_M_F \t - SUCCESS")
            return [myrand.choice(dist_answerers)]
        if limit_male != 1:
            limit_male = int(float(tmp_n_answers) / (limit_male + limit_female) * limit_male)
        if limit_male == len(dist_answerers) and limit_female == 1:
            limit_male -= 1
        else:
            limit_female = round(float(tmp_n_answers) - limit_male)
    # handle if there are not enough elements in the list
    for i in dist_answerers:
        if gen_participants[i] == "M":
            real_limit_male += 1
        elif gen_participants[i] == "F":
            real_limit_female += 1
    if gender_fixed_quantity != True: 
        diff_female = real_limit_female - limit_female
        diff_male = real_limit_male - limit_male
        if diff_male < 0: 
            limit_male = real_limit_male
            tmp_n_answers += diff_male
            logging.info(f"handle_M_F \t - WARNING: limit_male is too high. Value will be reduced.")
        if diff_female < 0: 
            limit_female = real_limit_female
            tmp_n_answers += diff_female
            logging.info(f"handle_M_F \t - WARNING: limit_female is too high. Value will be reduced.")
    else:
        if (real_limit_female - limit_female) < 0: 
            raise Exception("handle_M_F2 - Not enough female Files in the folder")
        if (real_limit_male - limit_male) < 0: 
                raise Exception("handle_M_F2 - Not enough male Files in the folder")
    # create answerers array
    answerers = []
    while True:
        random.shuffle(dist_answerers)
        M_F_selector = ["M", "F"]
        random.shuffle(M_F_selector)
        for _ in range(2):
            if len(dist_answerers) == 0 or len(answerers) == tmp_n_answers:
                logging.info(f"handle_M_F \t - SUCCESS")
                return answerers
            if limit_male == 0 and limit_female == 0:
                logging.info(f"handle_M_F \t - SUCCESS")
                return answerers
            if (M_F_selector[0] == "M" and limit_male > 0) or (M_F_selector[0] == "F" and limit_female > 0) :
                for i in range(len(dist_answerers)):
                    if gen_participants[dist_answerers[i]] == M_F_selector[0]:
                        answerers.append(dist_answerers[i])
                        dist_answerers.pop(i)
                        if M_F_selector[0] == "M": limit_male -= 1
                        else: limit_female -= 1
                        break

def search_person(matrice:list, person:str, tmp_volume:str, tmp_question:str):
    '''Search the person related to the question with a specific volume inside the matrix'''
    for i in matrice:
        if person == str(i[1]) and tmp_question == i[2]:
            if volume != "ND" or (get_volume(i[0]) == tmp_volume):                
                logging.info(f"search_person \t\t - SUCCESS")
                return i[0]
    logging.info(f"search_person \t\t - ERROR: No found file n:{tmp_question}, person:{person}, vol:{tmp_volume}")
    return None

def dialogs_handler(dir_path:str):
    '''CREATE FILE_NAMES\n\n
    Core function of the programm: \n
    creates a dictionary of random file names from your chosen folder'''
    _, count_a, count_q, initial_questions, _, a_letters, q_letters = folder_info(os.path.join(dir_path, input_folder))
    logging.info(f"dialogs_handler - \t - INFO: {count_a, count_q, initial_questions}")
    # create 3D array for questions
    matr_questions = list_to_3Dlist(count_q)
    matr_answers = list_to_3Dlist(count_a)
    matr_initquest = list_to_3Dlist(initial_questions)
    q_participants, a_participants = participants_lists(q_letters, a_letters)

    # handle 1 element arrays
    if len(a_participants) == len(q_participants) == 1 and a_participants[0] == q_participants[0]:
        raise Exception("More than 1 participant are required for the experiment!!")
    if len(a_participants) == 1:
        answerer = a_participants[0]
        # first interrogator must be != answerer
        if answerer in q_participants:
            q_participants.remove(answerer)
    if len(q_participants) == 1:
        interrogator = q_participants[0]
        # first answerer must be != interrogator
        if interrogator in a_participants:
            a_participants.remove(interrogator)

    participants = merge_arrays(q_participants, a_participants)

    if volume == "ND":
        global pos_participants
        pos_participants = {item: myrand.choice([0, 1]) for item in participants}

    if limit_male_female != "0:0":
        limit_male, limit_female = check_limits()
        gen_participants = find_gender(participants)
        
    # max number of participants to answers
    list_questions = questions_shuffler(matr_questions)
    logging.info(f"dialogs_handler \t - INFO: list_questions: {list_questions}")
    dict_answers = matr_to_dict1(matr_questions, list_questions)
    logging.info(f"dialogs_handler \t - INFO: dict_answers: {dict_answers}")

    tmp_n_answers = 0
    file_names = []
    first_question_enabler = False
    for j in list_questions:
        # choose random interrogator
        interrogator = myrand.choice(q_participants)
        answerers = list(set(dict_answers.get(j)))

        # handle number of answers also if negative 
        if n_answers == 0:
            tmp_n_answers = myrand.randint(1, len(answerers))
        elif n_answers < 0:
            tmp_n_answers = myrand.randint(1, min((abs(n_answers)-1), len(answerers)))
        else:
            tmp_n_answers = n_answers

        # if tmp_n_answers is 1 generate 2, but consider just one
        tmp_n_answers_2 = tmp_n_answers
        if tmp_n_answers == 1:
            tmp_n_answers_2 = 2

        tmp_cycle_limit = 10
        prev_answerers = answerers
        count_cycle_limit = 1
        # shuffle answerers 
        if limit_male_female == "0:0":
            while True:
                random.shuffle(prev_answerers)
                answerers = prev_answerers[:tmp_n_answers_2]
                if answerers[0] != interrogator:
                    logging.info(f"dialogs_handler \t - INFO: answerers {answerers}")
                    break
                elif count_cycle_limit > tmp_cycle_limit:
                    raise Exception(f"Internal Error: answerers: {answerers}, interrogator: {interrogator}")
                else:
                    count_cycle_limit +=1
        else:
            if gender_fixed_quantity == True:
                tmp_n_answers_2 = limit_male + limit_female
            while True:
                random.shuffle(answerers)
                answerers = handle_M_F(answerers, limit_male, limit_female, tmp_n_answers_2, gen_participants)
                if answerers[0] != interrogator:
                    tmp_n_answers_2 = len(answerers)
                    logging.info(f"dialogs_handler \t - INFO: answerers {answerers}")
                    break
                elif count_cycle_limit > tmp_cycle_limit:
                    raise Exception(f"Internal Error: answerers: {answerers}, interrogator: {interrogator}")
                else:
                    count_cycle_limit +=1
                    answerers = prev_answerers
        
        print(f"\t chosen question n: {j}", end=" ")

        tmp_volume = volume_handler(answerers[0], interrogator)

        # add first person to ask X each question
        tmp_questions = search_person(matr_questions, interrogator, tmp_volume, j)
        if tmp_questions != None:
            if first_question == True:
                file_names = add_file(file_names, tmp_questions)
            elif first_question == False:
                if first_question_enabler == False:
                    first_question_enabler = True
                else:
                    file_names = add_file(file_names, tmp_questions)
        print(f"with n{tmp_n_answers} answers from:", end=" ")
        #print(f"{interrogator}", end="-")
        for i_a in range(tmp_n_answers):
            if i_a != 0:
                added_q = 0
                tmp_volume = volume_handler(responder, answerers[i_a])
                if (myrand.random() < prob_prompt) == True:
                    tmp_initquest = search_person(matr_initquest, responder, tmp_volume, j)
                    if tmp_initquest != None:
                        file_names = add_file(file_names, tmp_initquest)
                        added_q += 1
                if myrand.random() < prob_question == True:
                    tmp_questions = search_person(matr_questions, responder, tmp_volume, j)
                    if tmp_questions != None:
                        file_names = add_file(file_names, tmp_questions)
                        added_q += 1
                if (added_q == 0) and (myrand.random() <= prob_p_q):
                    tmp_initquest = search_person(matr_initquest, responder, tmp_volume, j)
                    tmp_questions = search_person(matr_questions, responder, tmp_volume, j)
                    file_names = add_file(file_names, tmp_initquest)
                    file_names = add_file(file_names, tmp_questions)
                #print(f"{responder}", end="-")
            responder = answerers[i_a]
            tmp_answers = search_person(matr_answers, responder, tmp_volume, j)
            if tmp_answers != None:
                file_names = add_file(file_names, tmp_answers)
                print(answerers[i_a], end=" ")
        print("")
    logging.info(f"dialogs_handler \t - SUCCESS: {file_names}")
    return file_names

def dialogs_list(dir_path:str):
    '''Generates a dict for every file randomly chosen.\n
    If a custom settings was found, reads it and returns \n
    call handle_auto_files and saves the dict with every file name into __pycache__'''
    if custom_files_enabler:
        print("\t loading custom audio settings file...")
        return custom_files()
    else:
        print("\t chosing new files and pauses...")
        file_names = dialogs_handler(dir_path)
        with open((save_name1), 'w') as file:
            json.dump(file_names, file)
        if volume == "ND":
            save_name2 = os.path.splitext(save_name1)[0] + '_pos' + os.path.splitext(save_name1)[1]
            with open(save_name2, 'w') as file:
                json.dump(pos_participants, file)
    return file_names

def custom_files():
    ''' if custom file is a dict, return the dict, 
    if custom file is an array, find the file and check the dictionary.\n
    Handle the errors'''
    file_names = []
    try:
        with open(import_name1, "r") as file_json:
            data = json.load(file_json)
            if isinstance(data, list):
                if all(isinstance(item, dict) for item in data):
                    logging.info(f"custom_files \t - INFO: found dict")
                    file_names = data
                elif all(isinstance(item, str) for item in data):
                    logging.info(f"custom_files \t - INFO: found list")
                    tmp_input_folder = os.path.join(dir_path, input_folder)
                    max_participants, _, _, _, _, _, _ = folder_info(tmp_input_folder)
                    if len(data) > 1 and len(data) <= max_participants:
                        file_names = add_list_files(data, tmp_input_folder)
                    else:
                        raise Exception('file JSON does not contain a valid array. Format should be ["a.wav","b.wav",..]')
                else: 
                    raise Exception('file JSON does not contain a valid list')
            else: 
                raise Exception('file JSON does not contain a list')
        if volume == "ND":
            save_name2 = os.path.splitext(import_name1)[0] + '_pos' + os.path.splitext(import_name1)[1]
            with open(save_name2, 'r') as file_json:
                global pos_participants
                data2 = json.load(file_json)
                if isinstance(data2, dict):
                    logging.info(f"custom_files \t - INFO: found pos_participants file")
                    pos_participants = data2
                else:
                    raise Exception('file JSON does not contain a dictionary')
    except FileNotFoundError:
        if volume != "ND":
            raise Exception(f"file {import_name1} not found.")
        else:
            raise Exception(f"file {import_name1} or file {save_name2} not found.")
    logging.info(f"custom_files \t - SUCCESS: {file_names}")
    return file_names

def add_list_files(file_list:list, tmp_input_folder:str):
    """find each file in file_lisr in tmp_input_folder and
    create a file_names list"""
    for tmp_file in file_list:
        file = find_file(tmp_file, tmp_input_folder)
        file_names = add_file(file_names, file)
    logging.info(f"add_list_files \t\t - SUCCESS")
    return file_names

def write_files(OUTPUT:list):
    print("\t writing files into the hard drive", end="")
    output_path = os.path.join(dir_path, output_folder)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    point_counter = 0
    point_length = len(OUTPUT)
    for i in OUTPUT:
        point_counter += 1
        if point_counter % (point_length // 3) == 0: print(".", end="")
        if volume == "ND" and i[1] in pos_participants:
            write_name = os.path.join(output_path, f'merged{i[1]}_{pos_participants[i[1]]}.wav')
        else:
            write_name = os.path.join(output_path, f'merged{i[1]}_{volume}.wav')
        sf.write(write_name, i[0], sample_rate)
        logging.info(f"write_files \t\t - SUCCESS: Created {write_name}")
    logging.info(f"write_files \t\t - SUCCESS: All files Saved")
    print()
    print("\n\tCOMPLETED! (folder opened)")

if __name__ == '__main__':
    #try:
        print("\n\tGeneratore di dialoghi realistici.\n")
        file_names = dialogs_list(dir_path)
        '''Create output array [data, person] and silences/pauses values'''
        silences = silence_generator(file_names)
        '''create an array of pause_length for each (between) file'''
        OUTPUT = dialogs_join(file_names, silences)
        '''Create output array [data, person]: add silences/pauses to output data'''
        OUTPUT = burst(file_names, OUTPUT, silences)
        '''Write files to the hard drive'''
        write_files(OUTPUT)
        os.startfile(os.path.join(dir_path, output_folder))
    #except Exception as e:
    #    print(f"\n ! ERRORE ! \n\tOperation aborted due to internal error: {e}")
    #    exit()