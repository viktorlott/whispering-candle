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
docker run -v $(pwd):$(pwd) -w $(pwd) jrottenberg/ffmpeg:3.4-scratch -i $(pwd)/record.mp4 -ar 16000 $(pwd)/output.wav

# Right now you can test the repo by using this command
cargo run --release -- --input <file> --model <tiny|base>
```

### Thanks
Rewrite of https://github.com/huggingface/candle/tree/main/candle-examples/examples/whisper

Big thanks to [LaurentMazare](https://github.com/LaurentMazare)