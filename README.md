# Whispering candle

Under development.

### Safetensors
Whisper has partially converted their models into safetensors.

### Resources
- https://huggingface.co/docs/safetensors/index
- https://medium.com/@mandalsouvik/safetensors-a-simple-and-safe-way-to-store-and-distribute-tensors-d9ba1931ba04


### Commands
```sh
# If ffmpeg isn't installed on your system:
docker run -v $(pwd):$(pwd) -w $(pwd) jrottenberg/ffmpeg:3.4-scratch -i $(pwd)/record.wav -ar 16000 $(pwd)/output.wav
docker run -v $(pwd):$(pwd) -w $(pwd) jrottenberg/ffmpeg:3.4-scratch -i $(pwd)/src/assets/record.wav -f f32le -acodec pcm_f32le $(pwd)/output.pcmf32

# `-ar 16000` - audio sample rate
# `-ac 1` -  audio channels
# `-sample_fmt s16` - bit per sec
# `-ss 00:00:00 -t 00:01:00` - slice 1 min
docker run -v $(pwd):$(pwd) -w $(pwd) jrottenberg/ffmpeg:3.4-scratch -i $(pwd)/record.wav -ar 16000 -ac 1 $(pwd)/output.wav


# Right now you can test the repo by using this command
cargo run --release -- --input <file> --model <tiny|base>
```

### Thanks
Rewrite of https://github.com/huggingface/candle/tree/main/candle-examples/examples/whisper

Big thanks to [LaurentMazare](https://github.com/LaurentMazare)