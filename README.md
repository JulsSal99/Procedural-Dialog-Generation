<img src="/img/LIM.png" alt="drawing" width="200"/>

# Procedural Dialogue Generation

This is a thesis project carried out at the **Laboratory of Musical Informatics (LIM)**. The purpose of the thesis is to create an audio dataset from various participants in a soundproof room and then have these individual dialogues interact to construct a discourse involving multiple people. 

## Documentation
[Document](TESI.MD)

[Test Protocol](PROTOCOLLO.MD)

## The Program

In the following repository, you'll find a program for procedural dialogue generation. The program takes a series of input files and concatenates them. The user can then decide various parameters such as the presence of burst, the number of questions, which values should be random, etc.

[Python Pseudo-Realistic Dialogue Generator](PYGenerator.py)

The **input file** types include:
- Question (e.g., "How was your day?")
- Response (e.g., "Not much, it was okay.")
- Prompt (e.g., "And you?")
- Burst (e.g., cough)

The **output file** types are:
- Individual audio files for each speaker
- A combined file summarizing the previous tracks

## Settings

Users can declare various values for dialogue generation. If absent, default values are used.

[Settings](PYGenerator.cfg)


Below is an example of the configuration file code.

Comments provide explanations for each parameter.

Commands, when uncommented and marked with "Example:", represent the program's default values.

```cfg
[global]
# if True, question order is random
random_q_order = True

# number of questions. positive number, negative act as a threshold for random, 0 is just random QUANTITY
n_questions = 0

# number of answers. positive number, negative act as a threshold for random, 0 is just random
n_answers = 0

# Percentage initial question presence. 1 always, 0 never
prob_prompt = 0.5

# Percentage question presence. 1 always, 0 never
prob_question = 0.5

# probability an initial question will be followed by a new question
prob_p_q = 0.8

# volume of answers. "ND" if NOT DEFINED, "L" if LOW volume, "H" if HIGH volume
volume = ND

# decide if a dialogue will start with a question or an answer. It doesn't apply to each question, but just the first question
first_question = True


[gender] # Ratio number of male and female 
# if you want the output quantity to be this exact value, set True. Warning: it can cause errors if here are not enough participants in the folder
fixed_quantity = False

# proportion between male and female, in case there are not enough male or female to accomplish this task, the higher number will be reduced
male_female_ratio = 0:0


[files] # configuration/input/output files
# file name format: *IDname_SESSO_volume_tipo_ndomanda". eg. 01_M_H_A_01 The number identifies the position
name_format = person_gender_volume_type_question

# master folder. Should NOT end with a "/"
# Example: dir_path = C:/Users/giuli/Music/Edit

# You can add a background noise to the audio file that will be overlaid. 
# the noise name should be declared to use the noise. By default, the noise is in dir_path
# Example: noise_file = noise.wav

# input files folder inside master folder
# Example: input_folder = INPUT

# output files folder inside master folder
# Example: output_folder = OUTPUT 

# if custom_file is specified, it will take a dialogue user-generated order. See the end for details. The file is always inside the "custom" folder
# Example: custom_path = output_files.json

# if custom_sounds is specified, it will take a sounds list user-generated. See the end for details. The file is always inside the "custom" folder
# Example: custom_sounds = sounds.json

[pauses] # pauses values (in seconds)
min = 0.7
max = 2.0


[silences] # silences values (in seconds)
min = 0.05
max = 0.120


[long pauses] # long pauses (a pause after an answer) values (in seconds)
min = 0.9
max = 1.2


[sounds]
# This float value goes from 0 to 1. If 1, uses all sounds, if 0, none
s_quantity = 1

# minimum distance between the start of one sound and another. This DOES not consider answers, only sounds (in seconds)
min_s_distance = 5

# redundancy before and after to avoid overlap sounds (in seconds)
cut_redundancy = 1.5

# you can also specify how loud a sound should be if you want to
sound_amp_fact = 1

# how many times does the random function search for an empty space. Bigger values get better results, but a slower code
cycle_limit = 10

# if you don't want the same sound too close to another, just add this variable (in seconds)
similar_distance = 5


[fade] # apply fade in- and fade-out to each sample
# length of the fade. (in seconds)
fade_length = 0

# fade type. There are 2 values: 0 for logarithmic, 1 for linear 
fade_type = 0
```

custom_path file (position is inside the "temp" folder) shoud have one of these formats:
```json
["03_M_Q_01_L.wav", "01_M_A_01_L"]
```
```json
[{"path": "C:/Users/giuli/Music/Edit\\INPUT/03_M_Q_01_L.wav", "data": 0, "name": "03_M_Q_01_L", "person": "03", "duplicated": false}, {"path": "C:/Users/giuli/Music/Edit\\INPUT/01_M_A_01_L.wav", "data": 0, "name": "01_M_A_01_L", "person": "01", "duplicated": false}]
```

if you want to specify a different person position ("1" = far, "2" = near) for a generation with ND volume, you can create a file with the same custom_path name, but, it needs to end with "_pos.json".
Example: if custom_path is output_files.json, your positions file should be output_files_pos.json .
Inside that file, format should be:
```json
 {"01": 1, "02": 1, "03": 1, "04": 0}
```

custom_sounds file (position is inside the "temp" folder) shoud have one of these formats:
```json
["03_M_Q_01_L.wav", "01_M_A_01_L"]
```
```json
{"01_M_L_B_06.wav": 66.07703229845748, "03_M_L_B_08.wav": 41.9889395611619}
```

## debugging
Inside the "temp" folder you can find some examples and a logging.log file. This file shows the progress during the execution of the program, so you can easily find any issue or functionality inside the python code.