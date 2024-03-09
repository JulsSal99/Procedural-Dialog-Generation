import sys
sys.path.insert(0, 'libs')
import libsinstall
libsinstall.install_libraries()

import os
import soundfile as sf
import numpy as np
import random
import logging
import configparser
import logger
logger.logger()

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
prob_question = config.getfloat('global', 'prob_question', fallback=0.5)
prob_init_question = config.getfloat('global', 'prob_init_question', fallback=0.5)
prob_i_q = config.getfloat('global', 'prob_i_q', fallback=0.8)
volume = config.get('global', 'volume', fallback="ND")
first_question = config.getboolean('global', 'first_question', fallback=True)
# gender
gender_fixed_quantity = config.getboolean('gender', 'fixed_quantity', fallback=False)
limit_male_female = config.get('gender', 'male_female_ratio', fallback="0:0")
# files
name_format = (config.get('files', 'name_format')).split("_")
dir_path = config.get('files', 'dir_path', fallback=os.path.dirname(os.path.realpath(__file__)))
input_folder = config.get('files', 'input_folder', fallback="INPUT")
output_folder = config.get('files', 'output_folder', fallback="OUTPUT")
import_name1 = os.path.join("__pycache__", config.get('files', 'custom_path', fallback="output_files.json"))
# fade
fade_length = config.getfloat('fade', 'fade_length', fallback=0)
fade_type = config.getint('fade', 'fade_type', fallback=0)
# noise
enable_noise = config.getboolean('noise', 'enable_noise', fallback=False)
noise_file = config.get('noise', 'noise_file', fallback="")
# silences
s_min = config.getfloat('silences', 'min', fallback=0.05)
s_max = config.getfloat('silences', 'max', fallback=0.120)
# long pauses
lp_min = config.getfloat('long pauses', 'min', fallback=0.9)
lp_max = config.getfloat('long pauses', 'max', fallback=1.2)
# pauses
p_min = config.getfloat('pauses', 'min', fallback=0.7)
p_max = config.getfloat('pauses', 'max', fallback=0.9)
# sounds
s_quantity = config.getfloat('sounds', 's_quantity', fallback=0.5)
min_s_distance = config.getfloat('sounds', 'min_s_distance', fallback=5)
cut_redundancy = config.getfloat('sounds', 'cut_redundancy', fallback=1.5)
length_sounds = config.getfloat('sounds', 'length_sounds', fallback=2)
sound_amp_fact = config.getfloat('sounds', 'sound_amp_fact', fallback=1)
end_tollerance = config.getfloat('sounds', 'end_tollerance', fallback=3)
cycle_limit = config.getint('sounds', 'cycle_limit', fallback=10)

# data
sample_rate = config.getint('data', 'sample_rate', fallback=0)
channels = config.getint('data', 'sample_rate', fallback=0)
pop_tollerance = sample_rate * 1
save_name1 = os.path.join("__pycache__", "output_files.json")
pos_participants = {}

def cfg_check(count_answers, count_questions):
    if (n_answers>count_answers):
        raise Exception(f"cfg_check - n_answers in configuration file must be <= {count_answers} or -1 for random.")
    if (n_questions>count_questions):
        raise Exception(f"cfg_check - n_answers in configuration file must be <= {count_answers} or -1 for random.")
    if enable_noise == True and noise_file == "":
        raise Exception("cfg_check - noise file should not be empty")
    if volume != "ND" or volume != "H" or volume != "L":
         raise Exception("cfg_check - volume should be ND, H or L")
    if gender_fixed_quantity != False and limit_male_female == "0:0":
        raise Exception("cfg_check - fixed_quantity should not be False if limit_male_female is empty")
    logging.info(f"cfg_check \t - SUCCESS")

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

def audio_file(path: str, data: bin, name: str, person: str, duplicated: bool):
    file = {}
    file['path'] = path
    file['data'] = data
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
    '''Search for the file by its name with and without the extension'''
    '''and return the first file found with the exact path of the file.'''
    for root, _, files in os.walk(path):
        for file in files:
            if name in file and (file.lower().endswith(".wav")):
                logging.info(f"find_file \t\t - SUCCESS.")
                return os.path.join(root, file)
    raise Exception(f"File {name}.wav not found in {path}")

def folder_info(folder_path):
    '''count the number of audio files in a folder and split questions from answers'''
    max_files = 0
    count_q = [] #array with all questions
    count_a = [] #array with all answers
    count_iq = [] #array with all initial answer
    count_s = [] #array with all sounds
    q_letters = {} #count questions persons
    a_letters = {} #count answers persons
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            if len(filename.split('_')) > len(name_format):
                logging.info(f"folder_info \t\t - ABORT: {filename} name format is NOT correct. See manual for correct file naming.\n")
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
    return max_files, count_a, count_q, count_iq, count_s, a_letters, q_letters

def check_SR_CH(name, rate_temp, channels_temp):
    '''handle SampleRate and Channels Exceptions'''
    '''To avoid unuseful file reads,'''
    '''this function handle only Exceptions'''
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
    '''audio data to lenght. Works with STEREO and MONO'''
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
    sum = np.add(raw_noise[:raw_length], raw_file)
    if len(raw_noise[:raw_length]) != len(raw_file) or len(raw_file) != len(sum):
        raise Exception("INTERNAL ERROR: function 'noise', lenght does not match")
    logging.info(f"noise \t\t\t - SUCCESS.")
    return sum

def concatenate_fade(audio1, audio2, shape):
    '''Concatenate two audio files using a noise as a bandade (two fade-in, fade-out)'''
    fade = int(fade_length * sample_rate)
    if enable_noise:
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
    #fade_out = np.linspace(0.15, 1.05, fade)
    fade_out = shape_fixer(np.subtract(1.1, fade_out), shape)
    fade_in = np.flip(fade_out)
    FO_audio1 = audio1[-fade:] * fade_out
    FI_audio2 = audio2[:fade] *fade_in
    if enable_noise:
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
        raise Exception("INTERNAL ERROR: concatenate_fade function, lenght does not match")
    logging.info(f"concatenate_fade \t - SUCCESS")
    return OUTPUT

def concatenate(data1, data2, pause_length):
    '''join 2 audio files data1, data2 and adds a pause between them'''
    '''handles noises or fade-in/fade-out'''
    n_sample_silence = int(sample_rate * pause_length)
    shape = len(data1.shape)

    if shape > 1: #stereo
        silence = np.zeros((n_sample_silence, channels))
    else: #mono
        silence = np.zeros((n_sample_silence,))
    if enable_noise:
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
    print("\t adding pauses and silences...")
    silences = []
    length = len(file_names) - 1
    for i in range(length):
        if get_type(file_names[i]['name']) == "I":
            # if the pause is a SILENCE
            pause_length = random.uniform(s_min, s_max)
        elif get_type(file_names[i]['name']) == "A" and i != length:
            if get_nquestion(file_names[i]['name']) and get_nquestion(file_names[i+1]['name']):
                pause_length = random.uniform(lp_min, lp_max)  # seconds
            else:
                pause_length = random.uniform(p_min, p_max)  # seconds
        else:
            pause_length = random.uniform(p_min, p_max)  # seconds
        silences.append(pause_length)
    logging.info(f"silence_generator \t - SUCCESS.")
    return silences

def file_complete(file_names, silences):
    for j in range(len(file_names)):
        if j == 0:
            OUTPUT = file_names[0]['data']
        else:
            # add pause and join elements full or empty base on i value
            OUTPUT = concatenate(OUTPUT, file_names[j]['data'], silences[j - 1])
    logging.info(f"file_complete \t\t - SUCCESS.")
    return OUTPUT

def data_reader(file_names):
    '''Read each single audio file and add raw data to file_names'''
    '''and check sample rate and channels with check_SR_CH'''
    global channels, sample_rate
    if channels == 0 or sample_rate == 0:
        # Get sample rate
        data_temp, temp_rate = sf.read(file_names[0]['path'])
        # Get channels number
        channels += get_channels(data_temp)
        sample_rate += temp_rate

    for i in range(len(file_names)):
        file_names[i]['data'], rate_temp = sf.read(file_names[i]['path'])
        channels_temp = get_channels(file_names[i]['data'])
        check_SR_CH(file_names[i]['name'], rate_temp, channels_temp)
    logging.info(f"data_reader \t\t - SUCCESS.")

def dialogs_join(file_names:list, silences:list):
    ''' MAIN FUNCTION: create the ending file'''
    ''' create the class inside file_names and return to concatenate()'''
    ''' check channels, sample_rate'''

    # Add audio data and check sample rate and channels
    data_reader(file_names)
    
    print("\t creating data...")
    OUTPUT2 = []
    # i è l'elemento da stampare con i dati, mentre j è l'elemento attuale
    for i in range(len(file_names)):
        print_person = file_names[i]['person']
        if not file_names[i]['duplicated']:
            print_name = file_names[i]['person']
            for j in range(len(file_names)):
                if j == 0:
                    # gestisce il primo elemento e lo mette vuoto se non è lui
                    if file_names[j]['person'] == print_person:
                        OUTPUT = file_names[0]['data']
                    else:
                        if channels > 1:
                            OUTPUT = np.zeros((int(len(file_names[0]['data'])), channels))
                        else:
                            OUTPUT = np.zeros((int(len(file_names[0]['data'])),))
                        if enable_noise:
                            OUTPUT = noise(OUTPUT)
                else:
                    # aggiunge pausa e concatena elementi pieni o vuoti in base al valore di i.
                    if file_names[j]['person'] == print_person:
                        OUTPUT = concatenate(OUTPUT, file_names[j]['data'], silences[j - 1])
                    else:
                        if channels > 1:
                            file_silence = np.zeros((int(len(file_names[j]['data'])), channels))
                        else:
                            file_silence = np.zeros((int(len(file_names[j]['data'])),))
                        if enable_noise:
                            file_silence = noise(file_silence)
                        OUTPUT = concatenate(OUTPUT, file_silence, silences[j - 1])
            OUTPUT2.append([OUTPUT, print_name])
            logging.info(f"dialogs_join \t - SUCCESS for: {print_name}")
    OUTPUT = file_complete(file_names, silences)
    OUTPUT2.append([OUTPUT, "COMPLETE"])
    return OUTPUT2, silences


# /////////////////////////////////// SOUNDS //////////////////////////////////

def filenames_lenghts(file_names, silences):
    '''Create array for each output file with path, person, start and end'''
    '''also handle silences'''
    arr = []
    lengh_end, length_start = 0, 0
    i = 0
    for filename in file_names:
        len_file = raw_to_seconds(filename["data"])
        lengh_end = len_file + lengh_end
        arr.append([filename["path"], filename["person"], length_start, lengh_end])
        length_start = lengh_end
        if i != (len(file_names)-1):
            length_start = length_start + silences[i]
            lengh_end = lengh_end + silences[i]
            i += 1
    logging.info(f"filenames_lenghts \t - SUCCESS: {arr}")
    return arr

def handle_sounds(sound_files, file_names, max_duration, silences, tmp_participants):
    '''create 2D list of sounds. Each sound has a random position in seconds'''
    '''There are various values to setup the randomness'''
    OUTPUT = []
    tmp_arr = []
    count_sounds = 0
    audio = filenames_lenghts(file_names, silences)
    for _ in range(ceil(s_quantity)):
        random.shuffle(sound_files)
        for filename in sound_files:
            count_sounds += 1
            tmp_limit = 0
            person = get_person(filename)
            while True:
                if person not in tmp_participants:
                    correct = False
                    tmp_limit += 1
                    logging.info(f"handle_sounds \t\t - INFO: tmp_limit: {tmp_limit}: sound from a stranger")
                    break
                correct = True
                delay = random.uniform(0, max_duration-end_tollerance)
                if tmp_limit <= cycle_limit:
                    for i in tmp_arr:
                        # check if a sound is near another
                        if abs(i-delay) < min_s_distance:
                            correct = False
                            tmp_limit += 1
                            logging.info(f"handle_sounds \t\t - INFO: tmp_limit: {tmp_limit}: sound near another")
                            break
                        # WHAT-IF: The sound is near the same sound?
                    if correct != False:
                        for i_a in range(len(audio)):
                            start_NO_zone = audio[i_a][2]-float(cut_redundancy)-float(length_sounds)
                            start_NO_zone2 = audio[i_a][2]-float(cut_redundancy)
                            end_NO_zone = audio[i_a][3]+float(cut_redundancy)
                            if audio[i_a][1] == person and delay > (start_NO_zone) and delay < (end_NO_zone):
                                correct = False
                                tmp_limit += 1
                                logging.info(f"handle_sounds \t\t - INFO: tmp_limit: {tmp_limit}: sound is inside an audio of the same person")
                                break
                            """elif delay > (start_NO_zone2) and delay < (end_NO_zone):
                                print(delay, start_NO_zone2, end_NO_zone, audio[i_a][2], cut_redundancy)
                                correct = False
                                tmp_limit += 1
                                logging.info(f"handle_sounds \t\t - INFO: tmp_limit: {tmp_limit}: sound is near the start of any sound")
                                break"""
                else:
                    logging.info(f"handle_sounds \t\t - WARNING: no more space for new sounds with cycle repeated {tmp_limit}. Added {len(OUTPUT)} sounds")
                    print(f"\t added {count_sounds} sounds.")
                    return OUTPUT
                if correct:
                    tmp_arr.append(delay)
                    OUTPUT.append([os.path.join(dir_path, input_folder+"/"+filename), person, delay])
                    logging.info(f"handle_sounds \t\t - NOTE: appended {filename} on position {delay} with cycle repeated {tmp_limit}")
                    tmp_limit = 0
                    break
        logging.info(f"handle_sounds \t\t - NOTE: arr expanded with {len(OUTPUT)} sounds")
    print(f"\t\t added {len(OUTPUT)} sounds.")
    logging.info(f"handle_sounds \t\t - SUCCESS")
    # ho in uscita un array di tutti i file audio che vanno sovrapposti con il nome di ogni persona
    return OUTPUT

def sounds(file_names, audio_no_s, silences):
    '''core function called to add burst into the dialogue'''
    _, _, _, _, sound_files, _, _ = folder_info(os.path.join(dir_path, input_folder))
    output = []
    if s_quantity == 0:
        logging.info(f"sounds \t\t\t - ABORT: s_quantity = 0")
        output = audio_no_s
    else:
        # -1 takes the last file (COMPLETE)
        print("\t adding new sounds...", end=" ")
        max_duration = raw_to_seconds(audio_no_s[-1][0])
        tmp_participants = []
        for i in audio_no_s:
            tmp_participants.append(i[1])
        sounds = handle_sounds(sound_files, file_names, max_duration, silences, tmp_participants)
        for i in audio_no_s:
            if i[1] != "COMPLETE":
                sum = i[0]
                for j in sounds:
                    # if the person is the same
                    if i[1] == j[1]:
                        sound =sf.read(j[0])[0]
                        if sound_amp_fact != 1.0:
                            sound = sound * sound_amp_fact
                        start_sound = int(float(sample_rate)*j[2])
                        end_sound = len(sound)+start_sound
                        shape = len(sum.shape)
                        if enable_noise:
                            sum_tmp = concatenate_fade(sum[:start_sound], sound, shape)
                            sum = concatenate_fade(sum_tmp, sum[end_sound:], shape)
                        elif fade_length == 0:
                            if channels > 1: #stereo
                                sum_tmp = np.concatenate((sum[:start_sound-1], sound, sum[end_sound-1:]), axis=0)
                            else: #mono
                                sum = np.concatenate((sum_tmp[:start_sound-1], sound, sum[end_sound-1:]))
                        else:
                            sum_tmp = concatenate_fade(sum[:(start_sound-1)], sound, shape)
                            sum = concatenate_fade(sum_tmp, sum[(end_sound-1):], shape)
                limit = ceil(max_duration*sample_rate)
                if len(sum) > limit or len(sum) < limit-2:
                    raise Exception (f"INTERNAL ERROR: {i[1]} (with lenght: {len(sum)}) does not match {limit} length")
                output.append([sum, i[1]])
            elif i[1] == "COMPLETE":
                for j in output:
                    if j[1] != "COMPLETE":
                        if 'summed_data' not in locals():
                            summed_data = j[0]
                        else:
                            summed_data = np.add(summed_data, j[0])
                output.append([summed_data, "COMPLETE"])
            logging.info(f"sounds \t\t\t - SUCCESS for: {i[1]}")
    return output

# /////////////////////////////////// SOUNDS //////////////////////////////////

def participants_lists(q_letters:dict, a_letters:dict):
    q_participants = list(q_letters.keys())
    a_participants = list(a_letters.keys())
    logging.info(f"participants_lists \t - SUCCESS")
    return q_participants, a_participants

def list_to_3Dlist(list1:list):
    # create 3D list for dicts
    arr1 = []
    for filename in list1:
        person = get_person(filename)
        n_question = get_nquestion(filename)
        arr1.append([os.path.join(dir_path, input_folder+"/"+filename), person, n_question])
    logging.info(f"list_to_3Dlist \t\t - SUCCESS")
    return arr1

def questions_shuffler(matr1:list, value1:list):
    list1 = []
    for _, _, n in matr1:
        if n not in list1:
            list1.append(n)
    if random_q_order:
        random.shuffle(list1)
    if value1 == 0:
        list1 = list1[:(random.randint(1,len(list1)))]
    elif int(value1) < 0:
        list1 = list1[:(random.randint(1,abs(int(value1))))]
    else:
        list1 = list1[:(value1-1)]
    return list1

def matr_to_dict1(matr1:list, list1:list):
    dict1 = {}
    for j in list1:
        for _, p, n in matr1:
            if j == n:
                dict1.setdefault(n, []).append(p)
    return dict1

def volume_handler(p1:str, p2:str):
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
    return merged_array

def find_gender(participants:list):
    '''Search for the file by its name with and without the extension'''
    '''and return the first file found with the exact path of the file.'''
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
    real_limit_male = 0; real_limit_female = 0
    if gender_fixed_quantity != True: 
        if tmp_n_answers == 1:
            return [random.choice(dist_answerers)]
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
                return answerers
            if limit_male == 0 and limit_female == 0:
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
                logging.info(f"search_person \t - SUCCESS")
                return i[0]
    logging.info(f"search_person \t - ERROR: No found file")
    return None

def dialogs_handler(dir_path:str):
    '''CREATE FILE_NAMES'''
    '''Core function of the programm: '''
    '''creates a dictionary of random file names from your chosen folder'''
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
        pos_participants = {item: random.choice([0, 1]) for item in participants}

    if limit_male_female != "0:0":
        limit_male, limit_female = check_limits()
        gen_participants = find_gender(participants)
        
    # max number of participants to answers
    list_questions = questions_shuffler(matr_questions, n_questions)
    dict_answers = matr_to_dict1(matr_questions, list_questions)

    tmp_n_answers = 0
    file_names = []
    for j in list_questions:
        # choose random interrogator
        interrogator = random.choice(q_participants)
        answerers = dict_answers.get(j)
        answerers = list(set(answerers))

        # handle number of answers also if negative 
        if n_answers == 0:
            tmp_n_answers = random.randint(1, len(answerers))
        elif n_answers < 0:
            tmp_n_answers = random.randint(1, min((abs(n_answers)-1), len(answerers)))
        else:
            tmp_n_answers = n_answers

        if limit_male_female == "0:0":
            # shuffle answerers 
            while True:
                random.shuffle(answerers)
                if answerers[0] != interrogator:
                    break
            answerers = answerers[:tmp_n_answers]
        else:
            if gender_fixed_quantity == True:
                tmp_n_answers = limit_male + limit_female
            answerers = handle_M_F(answerers, limit_male, limit_female, tmp_n_answers, gen_participants)
            tmp_n_answers = len(answerers)
        
        print(f"\t chosen question n: {j}", end=" ")

        tmp_volume = volume_handler(answerers[0], interrogator)

        # add first person to ask X each question
        tmp_questions = search_person(matr_questions, interrogator, tmp_volume, j)
        if tmp_questions != None and first_question:
            file_names = add_file(file_names, tmp_questions)
        
        print(f"with n{tmp_n_answers} answers from:", end=" ")

        for i_a in range(tmp_n_answers):
            print(answerers[i_a], end=" ")
            err_check = 0
            if i_a != 0:
                tmp_volume = volume_handler(responder, answerers[i_a])
                if (random.random() < prob_init_question) == True:
                    tmp_initquest = search_person(matr_initquest, responder, tmp_volume, j)
                    if tmp_initquest != None:
                        file_names = add_file(file_names, tmp_initquest)
                        err_check += 1
                else:
                    err_check += 1
                if ((random.random() < prob_question) == True) and ((random.random() < prob_i_q) == True):
                    tmp_questions = search_person(matr_questions, responder, tmp_volume, j)
                    if tmp_questions != None:
                        file_names = add_file(file_names, tmp_questions)
                        err_check += 1
                else:
                    err_check += 1
            responder = answerers[i_a]
            tmp_answers = search_person(matr_answers, responder, tmp_volume, j)
            if tmp_answers != None:
                file_names = add_file(file_names, tmp_answers)
                err_check += 1
            if not (err_check == 3 or (i_a == 0 and err_check == 1)):
                logging.info(f"dialogs_handler \t - ERROR: {matr_answers}, responder: {responder}, j: {j}, i_a: {i_a}, {volume, tmp_volume}")
                raise Exception(f"Wrong Setting: Can't find the right file responder {responder}, question {j} and volume {tmp_volume}")
        print("")
    logging.info(f"dialogs_handler \t - SUCCESS: {file_names}")
    return file_names

def dialogs_list(dir_path:str):
    '''Generates a dict for every file randomly chosen'''
    '''if a custom settings was found, reads it and returns'''
    '''call handle_auto_files and saves the dict with every file name into __pycache__'''
    import json
    if os.path.exists(import_name1) and save_name1!=import_name1:
        print("\t found custom audio settings file...")
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
    ''' if custom file is a dict, return the dict, '''
    ''' if custom file is an array, find the file and check the dictionary'''
    ''' handle the errors'''
    import json
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
                    max_participants, _, _, _, _, _, _ = folder_info(os.path.join(dir_path, input_folder))
                    if len(data) > 1 and len(data) <= max_participants:
                        for tmp_file in data:
                            file = find_file(tmp_file, os.path.join(dir_path, input_folder))
                            file_names = add_file(file_names, file)
                    else:
                        raise Exception('file JSON does not contain a valid array. Format should be ["a.wav","b.wav",...]')
                else: 
                    raise Exception('file JSON does not contain a valid list')
            else: 
                raise Exception('file JSON does not contain a list')
        if volume == "ND":
            save_name2 = os.path.splitext(save_name1)[0] + '_pos' + os.path.splitext(save_name1)[1]
            with open(save_name2, 'r') as file_json:
                global pos_participants
                data2 = json.load(file_json)
                if all(isinstance(item2, dict) for item2 in data2):
                    logging.info(f"custom_files \t - INFO: found pos_participants file")
                    pos_participants = data2
    except FileNotFoundError:
        print(f"file {import_name1} not found.")
    logging.info(f"custom_files \t - SUCCESS: {file_names}")
    return file_names

def write_files(OUTPUT:list):
    print("\t writing files into the hard drive...")
    for i in OUTPUT:
        if volume == "ND" and i[1] in pos_participants:
            write_name = os.path.join(dir_path, output_folder+f'/merged{i[1]}_{pos_participants[i[1]]}.wav')
        else:
            write_name = os.path.join(dir_path, output_folder+f'/merged{i[1]}_{volume}.wav')
        sf.write(write_name, i[0], sample_rate)
        logging.info(f"write_files \t\t - SUCCESS: Created {write_name}")
    logging.info(f"write_files \t\t - SUCCESS: All files Saved")
    print("\n COMPLETED! (folder opened)")

if __name__ == '__main__':
    try:
        print("\n\tGeneratore di dialoghi realistici.\n")
        file_names = dialogs_list(dir_path)
        '''Create output array [data, person] and silences/pauses values'''
        silences = silence_generator(file_names)
        '''create an array of pause_length for each (between) file'''
        OUTPUT, silences = dialogs_join(file_names, silences)
        '''Create output array [data, person]: add silences/pauses to output data'''
        OUTPUT = sounds(file_names, OUTPUT, silences)
        write_files(OUTPUT)
        os.startfile(os.path.join(dir_path, output_folder))
    except Exception as e:
        print(f"\n ! ERRORE ! \n\tOperation aborted due to internal error: {e}")
        exit()