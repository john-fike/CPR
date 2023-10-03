inputFile = "test.txt"

with open(inputFile, 'r') as input:
    chunkNumber = 0
    while True:
        chunk = input.read(4000)
        if not chunk:
            break
        outputFile = str(chunkNumber) + ".txt"
        with open(outputFile, 'w') as output:
            output.write(chunk)
        chunkNumber += 1