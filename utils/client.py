# pip install websocket_client
import threading
import time
import sys
import time
import json
import websocket
import threading
from queue import Queue
import argparse
import os


def send_data(test_audio, ws, audio_sleep, step, send_data_flag_queue):
    send_time = 0
    cnt = 1
    with open(test_audio, 'rb') as f:
        while True:
            binary_data = f.read(step)
            if binary_data:
                try:
                    print(len(binary_data))
                    ws.send(binary_data, websocket.ABNF.OPCODE_BINARY)
                    # sys.stdout.write("[ info ]: send count:%s\tsend time:%s\n"% (cnt,time.time()))
                    cnt += 1
                except:
                    sys.stdout.write(
                        '[ info ]: send audio_frame >>> socket closed]\n')
                    break
            if len(binary_data) < step:
                break
            time.sleep(audio_sleep)
            send_time += audio_sleep * 1000
    try:
        content_str = json.dumps(content_end)
        ws.send(content_str, websocket.ABNF.OPCODE_TEXT)
    except:
        sys.stdout.write('[ info ]: send closed_frame >>> socket closed\n')

    # sys.stdout.write('[ info ]: send_data_flag set True\n')
    send_data_flag_queue.put(True)


def rec_data(test_audio, ws, rec_data_flag_queue):
    # sys.stdout.write(f"{test_audio}\n")
    # opcode, data = ws.recv_data()
    data = ws.recv()
    # data = data.decode('utf-8')  # b'\x03\xeaIllegal status code: 1006'
    if 'emotion' in data:
        data = json.loads(data)
        data['wav'] = test_audio
    sys.stdout.write(f"{data}\n")
    rec_data_flag_queue.put(True)


def test(test_audio, audio_sleep, step):
    url = "ws://103.177.28.71:20789"  # ws://speech.sensetime.com/offline-ser; ws://0.0.0.0
    # url = "ws://172.20.52.195:20789"  # ws://speech.sensetime.com/offline-ser; ws://0.0.0.0
    # url = "ws://0.0.0.0:" + args.port  # ws://speech.sensetime.com/offline-ser; ws://0.0.0.0
    # url = "wss://speech.sensetime.com/offline-ser"  # ws://speech.sensetime.com/offline-ser; ws://0.0.0.0

    # sys.stdout.write("################################################################################\n")
    # sys.stdout.write("[ info ]: start test\n")
    # sys.stdout.write("[ info ]: audio_sleep %f\n" % audio_sleep)

    # sys.stdout.write("[ info ]: test_url %s\n"%url)
    try:
        ws = websocket.create_connection(url)
    except Exception as e:
        print('[ error ]: connect url error:%s' % e)
        ws.close()
    content_str = json.dumps(content_start)

    ws.send(content_str, websocket.ABNF.OPCODE_TEXT)

    rec_data_flag_queue = Queue(maxsize=1)
    send_data_flag_queue = Queue(maxsize=1)

    send_thread = threading.Thread(target=send_data, args=(
        test_audio, ws, audio_sleep, step, send_data_flag_queue))
    rec_thread = threading.Thread(target=rec_data, args=(
        test_audio, ws, rec_data_flag_queue))
    send_thread.setDaemon(True)
    rec_thread.setDaemon(True)
    rec_thread.start()
    send_thread.start()


    while 1:
        if send_data_flag_queue.empty() == True:
            time.sleep(audio_sleep)
        else:
            # sys.stdout.write("[ info ]: thread-send complete, send_data_flag_queue:%s\n"%send_data_flag_queue.get())
            break

    time_spread = 0

    while 1:
        if rec_data_flag_queue.empty() == False or time_spread > 30:
            if time_spread > 30:
                sys.stdout.write('[ info ]: closed_time > 30s\n')
            # else:
                # sys.stdout.write("[ info ]: thread-recv complete, rec_data_flag_queue:%s\n"%rec_data_flag_queue.get())
            break
        else:
            pass
        time.sleep(0.1)
        time_spread += 0.1

    ws.close()
    # sys.stdout.write("[ info ]: success\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--audio_file', default='/home/xxx/cv/mer2023/dataset-process/features_test2/resnet50face_FRA_features_test2.tar.gz', help='测试集')

    parser.add_argument('-i', '--audio_sleep', default='0.00', help='发送时间')
    parser.add_argument('-s', '--step', default='128000', help='发送音频大小')  # 25600(best), 10240
    parser.add_argument('-c', '--process_num', default='1', help='并发数')
    parser.add_argument('-t', '--audio_type', default='pcm', help='音频格式')
    parser.add_argument('-r', '--samplerate', default='16000', help='采样率(16K)')
    parser.add_argument('-p', '--port', default='16578', help='端口号')
    args = parser.parse_args()

    audio_type = args.audio_type
    samplerate = int(args.samplerate)
    audio_file = args.audio_file
    filename = os.path.basename(audio_file)
    content_start = {"sampleRate": 16000, "channel": 1, "signal": "start", "filename": filename}
    content_start["audio_type"] = audio_type
    content_end = {"signal": "end"}

    process_num = args.process_num

    step = int(args.step)

    audio_sleep = float(args.audio_sleep)

    test(audio_file, audio_sleep, step)
