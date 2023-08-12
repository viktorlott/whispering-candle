use serde::Deserialize;

// The names in comments correspond to the original implementation:
// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L17
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub num_mel_bins: usize,            // n_mels
    pub max_source_positions: usize,    // n_audio_ctx
    pub d_model: usize,                 // n_audio_state
    pub encoder_attention_heads: usize, // n_audio_head
    pub encoder_layers: usize,          // n_audio_layer
    pub vocab_size: usize,              // n_vocab
    pub max_target_positions: usize,    //  n_text_ctx
    // pub n_text_state: usize,
    pub decoder_attention_heads: usize, // n_text_head
    pub decoder_layers: usize,          // n_text_layer
    pub suppress_tokens: Vec<u32>,
    // NOTE: Add only when necessary (https://huggingface.co/openai/whisper-large-v2/blob/main/config.json)
    // pub _name_or_path: String,
    // pub activation_dropout: f64,
    // pub activation_function: String,
    // pub architectures: Vec<String>,
    // pub attention_dropout: f64,
    // pub begin_suppress_tokens: Vec<u32>,
    // pub bos_token_id: usize,
    // pub d_model: usize,
    // pub decoder_attention_heads: usize,
    // pub decoder_ffn_dim: usize,
    // pub decoder_layerdrop: f64,
    // pub decoder_layers: usize,
    // pub decoder_start_token_id: usize,
    // pub dropout: f64,
    // pub encoder_attention_heads: usize,
    // pub encoder_ffn_dim: usize,
    // pub encoder_layerdrop: f64,
    // pub encoder_layers: usize,
    // pub eos_token_id: usize,
    // pub forced_decoder_ids: Vec<Vec<usize>>,
    // pub init_std: f64,
    // pub is_encoder_decoder: bool,
    // pub max_length: usize,
    // pub max_source_positions: usize,
    // pub max_target_positions: usize,
    // pub model_type: String,
    // pub num_hidden_layers: usize,
    // pub num_mel_bins: usize,
    // pub pad_token_id: usize,
    // pub scale_embedding: bool,
    // pub suppress_tokens: Vec<u32>,
    // pub torch_dtype: String,
    // pub transformers_version: String,
    // pub use_cache: bool,
    // pub vocab_size: usize,
}

impl Config {
    #[allow(dead_code)]
    pub fn tiny_en() -> Self {
        // List of tokens to be suppressed during the decoding process.
        // Extracted from the Whisper project's _get_suppress_tokens function.
        // These tokens might represent non-speech elements, irrelevant characters, etc.
        // From the _get_suppress_tokens function + 50362 (no timestamp)
        // https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/decoding.py#L605
        let suppress_tokens = vec![
            1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93,
            357, 366, 438, 532, 685, 705, 796, 930, 1058, 1220, 1267, 1279, 1303, 1343, 1377, 1391,
            1635, 1782, 1875, 2162, 2361, 2488, 3467, 4008, 4211, 4600, 4808, 5299, 5855, 6329,
            7203, 9609, 9959, 10563, 10786, 11420, 11709, 11907, 13163, 13697, 13700, 14808, 15306,
            16410, 16791, 17992, 19203, 19510, 20724, 22305, 22935, 27007, 30109, 30420, 33409,
            34949, 40283, 40493, 40549, 47282, 49146, 50257, 50357, 50358, 50359, 50360, 50361,
            50362,
        ];

        Self {
            num_mel_bins: 80,
            vocab_size: 51864,
            max_source_positions: 1500,
            d_model: 384,
            encoder_attention_heads: 6,
            encoder_layers: 4,
            max_target_positions: 448,
            // n_text_state: 384,
            decoder_attention_heads: 6,
            decoder_layers: 4,
            suppress_tokens,
        }
    }
}
