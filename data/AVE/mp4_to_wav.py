import os


train_videos = '/home/xxx/cv/MML/AVE_Dataset/video_list.txt'

train_audio_dir = '/home/xxx/cv/MML/AVE_proessed/AVE/audio'


# train set processing
with open(train_videos, 'r') as f:
    files = f.read().splitlines()

for i, item in enumerate(files):

    mp4_filename = os.path.join('/home/xxx/cv/MML/AVE_Dataset/AVE', item)
    wav_filename = os.path.join(train_audio_dir, item[:-4]+'.wav')

    print('ffmpeg -y -i {} -acodec pcm_s16le -ar 16000 -ac 1 {}'.format(mp4_filename, wav_filename))

