import whisper_openAI.whisper as whisper # version: 20230314
import numpy
import time
import pprint
random_temprature = numpy.random.randint(70,81)/100
model = whisper.load_model("large-v2")

audio = whisper.load_audio("xxx/spiritlm/output2.wav")
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio)
mel = mel.to('cuda')


for i in [2, 4, 8, 16, 32, 64, 128]:
    options = whisper.DecodingOptions(fp16 = True, without_timestamps = True, temperature=random_temprature,best_of=i)
    x_time = []
    x_res = []

    for turn in range(10):
        stime = time.time()
        result,_ = whisper.decode(model[0], mel, options)
        etime = time.time()
        x_res.append(result)
        x_time.append(etime-stime)

    #for idx, res in enumerate(result):
    #    print(f"Candiate {idx}: {res}")
    print(x_res)
    print('n_best:{}'.format(i))
    print('elapse time:{}'.format(sum(x_time)/len(x_time)))
    print('-'*100)

# -------------------------- #
'''
n_best | elapse time(sec) (average time for ten executions)
--------------------------------
2      | 1.9413817644119262
4      | 1.7820977210998534
8      | 1.9076156616210938
16     | 2.7661774158477783
32     | 5.087803649902344
64     | 9.676345181465148
128    | 18.842384457588196
'''
