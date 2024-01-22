import os


train_videos = '/home/xxx/cv/MML/k31/video_train_list.txt'
test_videos = '/home/xxx/cv/MML/k31/video_test_list.txt'

train_audio_dir = '/home/xxx/cv/MML/k31/audio_train'
test_audio_dir = '/home/xxx/cv/MML/k31/audio_test'


# test set processing
with open(test_videos, 'r') as f:
    files = f.readlines()

for i, item in enumerate(files):

    mp4_filename = os.path.join('/home/xxx/cv/MML/k31/video_test', item[:-1])
    wav_filename = os.path.join(test_audio_dir, item[:-5]+'.wav')
    if os.path.exists(wav_filename):
        pass
    else:
        print('ffmpeg -y -i {} -acodec pcm_s16le -ar 16000 -ac 1 {}'.format(mp4_filename, wav_filename))


# train set processing
with open(train_videos, 'r') as f:
    files = f.readlines()

for i, item in enumerate(files):

    mp4_filename = os.path.join('/home/xxx/cv/MML/k31/video_train', item[:-1])
    wav_filename = os.path.join(train_audio_dir, item[:-5]+'.wav')
    if os.path.exists(wav_filename):
        pass
    else:
        print('ffmpeg -y -i {} -acodec pcm_s16le -ar 16000 -ac 1 {}'.format(mp4_filename, wav_filename))





