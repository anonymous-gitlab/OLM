# *_*coding:utf-8 *_*
import os

############ For LINUX ##############
DATA_DIR = {
	'MER2023': '/home/xxx/cv/mer2023/dataset-process',
}
PATH_TO_RAW_AUDIO = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'audio'),
}
PATH_TO_RAW_FACE = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'openface_face'),
}

PATH_TO_RAW_AUDIO_TEST3 = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'test3_audio'),
}
PATH_TO_RAW_AUDIO_TEST1 = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'test1_audio'),
}
PATH_TO_RAW_AUDIO_TEST2 = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'test2_audio'),
}

PATH_TO_HOG = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'openface_hog'),
}
PATH_TO_POSE = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'openface_pose'),
}
PATH_TO_TRANSCRIPTIONS = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'transcription_test3.csv'),
}

PATH_TO_FEATURES = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'features'),
}
PATH_TO_FEATURES_TEST3 = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'features_test3'),
}
PATH_TO_FEATURES_TEST1 = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'features_test1'),
}
PATH_TO_FEATURES_TEST2 = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'features_test2'),
}

PATH_TO_FEATURES_PRETRAINED = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'features_pretrained'),
}
PATH_TO_LABEL = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'label-6way.npz'),
}

PATH_TO_PRETRAINED_MODELS = '/home/xxx/cv/mer2023/tools'
PATH_TO_OPENSMILE = '/home/xxx/cv/mer2023/tools/opensmile-2.3.0/'
PATH_TO_FFMPEG = '/home/xxx/cv/mer2023/tools/ffmpeg-4.4.1-i686-static/ffmpeg'
PATH_TO_NOISE = '/home/xxx/cv/mer2023/tools/musan/audio-select'

SAVED_ROOT = os.path.join('./saved')
DATA_DIR = os.path.join(SAVED_ROOT, 'data')
MODEL_DIR = os.path.join(SAVED_ROOT, 'model')
LOG_DIR = os.path.join(SAVED_ROOT, 'log')
PREDICTION_DIR = os.path.join(SAVED_ROOT, 'prediction')
FUSION_DIR = os.path.join(SAVED_ROOT, 'fusion')
SUBMISSION_DIR = os.path.join(SAVED_ROOT, 'submission')


############ For Windows (openface-win) ##############
DATA_DIR_Win = {
	'MER2023': 'Multimedia-Transformer\\MER2023-Baseline-master\\dataset-process',
}

PATH_TO_RAW_FACE_Win = {
	'MER2023':   os.path.join(DATA_DIR_Win['MER2023'],   'video'),
}

PATH_TO_FEATURES_Win = {
	'MER2023':   os.path.join(DATA_DIR_Win['MER2023'],   'features'),
}

PATH_TO_OPENFACE_Win = "Multimedia-Transformer\\MER2023-Baseline-master\\tools\\openface_win_x64"