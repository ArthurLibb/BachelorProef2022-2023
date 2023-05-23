from jiwer import wer

def WERModellen(dataset):
  listWERWisper = []
  listWERWav2Vec = []
  for audio in dataset['train']:
    transcription = audio['transcription']
    path = audio['audio']['path']
    #Wisper
    audio = whisper.load_audio(path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(modelWhisper.device)
    options = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(modelWhisper, mel, options)
    rate = wer(result.text, transcription)
    listWERWisper.append(rate)
    #Wav2Vec
    path = [path]
    modelTranscr = modelWav2Vec.transcribe(path)
    transW2V = modelTranscr[0]['transcription']
    rate2 = wer(transW2V,transcription)
    listWERWav2Vec.append(rate2)
  return gemiddelde(listWERWav2Vec),gemiddelde(listWERWisper), listWERWav2Vec,listWERWisper