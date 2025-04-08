'''
load whisper -> load llama -> whisper output -> 讓 llama 修正
'''

import torch
import torch.nn.functional as F
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from lit_llama import LLaMA, Tokenizer
from lit_llama.utils import EmptyInitOnDevice


def load_whisper_model():
    """加载Whisper模型和处理器"""
    #print("正在加载Whisper模型...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    return processor, model


def load_llama_model(checkpoint_path, tokenizer_path):
    """加载Lit-LLaMA模型和分词器"""
    #print("正在加载Lit-LLaMA模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with EmptyInitOnDevice(device=device, dtype=torch.float16):
        model = LLaMA.from_name("7B")

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # 加载分词器
    tokenizer = Tokenizer(tokenizer_path)

    return model, tokenizer


# 移除 Whisper 特殊 token
def clean_transcription(text):
    import re
    return re.sub(r"<\|.*?\|>", "", text).strip()



def transcribe_audio(audio_path, audio_processor, audio_model):
    """處理語音檔案"""
    speech, rate = sf.read(audio_path)

    if rate != 16000:
        speech = librosa.resample(speech, orig_sr=rate, target_sr=16000)
        rate = 16000

    if len(speech.shape)>1 and speech.shape[1] > 1:
        speech = librosa.to_mono(speech)

    input_values = audio_processor(speech, return_tensors="pt", sampling_rate=rate).input_features

    # Decoding
    with torch.no_grad():
        predicted_ids = audio_model.generate(input_values)

    transcription = audio_processor.batch_decode(predicted_ids)[0]
    return transcription

def correct_transcription(llama_model, tokenizer, transcription):
    """ 使用 llama 修正"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transcription = clean_transcription(transcription)
    #prompt = f"Correct the following ASR transcript:\n\n{transcription}\n\nCorrected:"
    prompt = f"Correct the following ASR transcript according to the context and meaning:\n\n{transcription}\n\nCorrected:"

    print(prompt)
    encoded_prompt = tokenizer.encode(prompt, bos=True, eos=True) #TODO: compare bos & eos

    if isinstance(encoded_prompt, torch.Tensor):
        prompt_tensor = encoded_prompt.clone().detach().to(device).unsqueeze(0)
    else:
        prompt_tensor = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0)


    # 生成修正文本
    max_new_tokens = 200
    generation_length = 0
    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = llama_model(prompt_tensor if not generated_tokens else torch.cat([prompt_tensor, torch.tensor([generated_tokens], dtype=torch.long, device=device)], dim=1))
            logits = logits[:, -1, :]

            #next_token = torch.argmax(logits, dim=-1).item() #argmax
            ### 使用 top_k sampling
            temperature = 0.8
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            generation_length += 1
            generated_tokens.append(next_token)

            if next_token == tokenizer.eos_id:
                break

    #corrected_text = tokenizer.decode(generated_tokens)
    corrected_text = tokenizer.decode(torch.tensor(generated_tokens, dtype=torch.long))
    return corrected_text


def main():
    audio_path = "/share/home/annaliang/Whispering-LLaMA/gigaspeech_row25_facebook.wav"
    llama_checkpoint_path = "/share/home/annaliang/Whispering-LLaMA/weight/alpaca.pth"
    tokenizer_path = "/share/home/annaliang/Whispering-LLaMA/weight/tokenizer.model"

    llama_model, llama_tokenizer = load_llama_model(llama_checkpoint_path, tokenizer_path)
    whisper_processor, whisper_model = load_whisper_model()

    transcription = transcribe_audio(audio_path, whisper_processor, whisper_model)
    corrected_text = correct_transcription(llama_model, llama_tokenizer, transcription)

    print(f"Whisper 轉錄结果: {transcription}")
    print(f"LLaMA 修正结果: {corrected_text}")

if __name__ == "__main__":
    main()
