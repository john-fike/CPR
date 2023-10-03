import whisper

inputSpeech = "./meetings/advisor_9-14-23.mp3"
outputText = inputSpeech + ".txt"

model = whisper.load_model("medium")
result = model.transcribe(inputSpeech)

with open(outputText, 'w') as output:
    output.write(result)