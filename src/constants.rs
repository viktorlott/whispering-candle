use candle::DType;

//----------------------------------------
// DATA TYPE CONFIGURATION
//----------------------------------------

// This constant defines the data type used for audio processing in this module.
// Here, F32 refers to the 32-bit floating-point number representation.
pub const DTYPE: DType = DType::F32;

//----------------------------------------
// AUDIO PARAMETERS
//----------------------------------------

// The sample rate represents the number of samples of audio carried per second, measured in Hz or kHz.
// Here, 16000 means that audio is sampled 16,000 times per second.
pub const SAMPLE_RATE: usize = 16_000;

// The number of bins to use for the Fast Fourier Transform. Essentially,
// this controls the size of the window on which the FFT will be performed.
pub const N_FFT: usize = 400;

// Number of Mel bands to generate. Mel bands are a series of triangular filters
// that simulate the frequency resolution of the human ear.
pub const N_MELS: usize = 80;

// The step size (in samples) between successive frames. Also known as stride.
// It determines how much the window moves on each iteration. For instance,
// a hop length of 160 on a sample rate of 16,000 means the window is moved every 10ms.
pub const HOP_LENGTH: usize = SAMPLE_RATE / 100;

// Duration of each chunk of audio processed at a time, measured in seconds.
// Here, each chunk is 30 seconds long.
pub const CHUNK_LENGTH: usize = 30;

// Total number of samples in a 30-second audio chunk.
// Derived by multiplying the chunk length by the sample rate.
pub const N_SAMPLES: usize = CHUNK_LENGTH * SAMPLE_RATE;

// Total number of frames present in the mel spectrogram input.
// Derived by dividing the number of samples by the hop length.
pub const N_FRAMES: usize = N_SAMPLES / HOP_LENGTH;

//----------------------------------------
// MODEL-RELATED CONSTANTS
//----------------------------------------

// Threshold for determining the presence of speech. If the calculated value
// exceeds this threshold, the audio segment might be considered as having no speech.
pub const NO_SPEECH_THRESHOLD: f64 = 0.6;

// Log probability threshold for filtering model predictions.
pub const LOGPROB_THRESHOLD: f64 = -1.0;

// Different temperature values for adjusting model prediction diversity.
// Higher values make the output more random, while lower values make it more deterministic.
pub const TEMPERATURES: [f64; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

// Threshold for the compression ratio used in audio processing or model predictions.
pub const COMPRESSION_RATIO_THRESHOLD: f64 = 2.4;

//----------------------------------------
// TOKEN-RELATED CONSTANTS
//----------------------------------------

// Token representing the start of a transcript or text segment.
pub const SOT_TOKEN: &str = "<|startoftranscript|>";

pub const TRANSCRIBE_TOKEN: &str = "<|transcribe|>";

// Token representing the end of a transcript or text segment.
pub const EOT_TOKEN: &str = "<|endoftext|>";

// Token representing segments where no speech is detected.
pub const NO_SPEECH_TOKEN: &str = "<|nocaptions|>";
