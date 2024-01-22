import os


train_videos = '/home/xxx/cv/MML/VGGSound_processed/train_video_list.txt'
test_videos = '/home/xxx/cv/MML/VGGSound_processed/test_video_list.txt'

train_audio_dir = '/home/xxx/cv/MML/VGGSound_processed/train-audios/train-set'
test_audio_dir = '/home/xxx/cv/MML/VGGSound_processed/test-audios/test-set'


# test set processing
with open(test_videos, 'r') as f:
    files = f.readlines()

for i, item in enumerate(files):

    mp4_filename = os.path.join('/home/xxx/cv/MML/VGGSound/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video', item[:-1])
    wav_filename = os.path.join(test_audio_dir, item[:-5]+'.wav')
    if os.path.exists(wav_filename):
        pass
    else:
        print('ffmpeg -y -i {} -acodec pcm_s16le -ar 16000 -ac 1 {}'.format(mp4_filename, wav_filename))


# train set processing
with open(train_videos, 'r') as f:
    files = f.readlines()

for i, item in enumerate(files):

    mp4_filename = os.path.join('/home/xxx/cv/MML/VGGSound/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video', item[:-1])
    wav_filename = os.path.join(train_audio_dir, item[:-5]+'.wav')
    if os.path.exists(wav_filename):
        pass
    else:
        print('ffmpeg -y -i {} -acodec pcm_s16le -ar 16000 -ac 1 {}'.format(mp4_filename, wav_filename))





