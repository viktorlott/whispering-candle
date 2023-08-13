use candle::{IndexOp, Result, Tensor, D};
use tokenizers::Tokenizer;

use super::constants;
use super::model::Whisper;
use super::utils::token_id;

const LANGUAGES: [(&str, &str); 99] = [
    ("en", "english"),
    ("zh", "chinese"),
    ("de", "german"),
    ("es", "spanish"),
    ("ru", "russian"),
    ("ko", "korean"),
    ("fr", "french"),
    ("ja", "japanese"),
    ("pt", "portuguese"),
    ("tr", "turkish"),
    ("pl", "polish"),
    ("ca", "catalan"),
    ("nl", "dutch"),
    ("ar", "arabic"),
    ("sv", "swedish"),
    ("it", "italian"),
    ("id", "indonesian"),
    ("hi", "hindi"),
    ("fi", "finnish"),
    ("vi", "vietnamese"),
    ("he", "hebrew"),
    ("uk", "ukrainian"),
    ("el", "greek"),
    ("ms", "malay"),
    ("cs", "czech"),
    ("ro", "romanian"),
    ("da", "danish"),
    ("hu", "hungarian"),
    ("ta", "tamil"),
    ("no", "norwegian"),
    ("th", "thai"),
    ("ur", "urdu"),
    ("hr", "croatian"),
    ("bg", "bulgarian"),
    ("lt", "lithuanian"),
    ("la", "latin"),
    ("mi", "maori"),
    ("ml", "malayalam"),
    ("cy", "welsh"),
    ("sk", "slovak"),
    ("te", "telugu"),
    ("fa", "persian"),
    ("lv", "latvian"),
    ("bn", "bengali"),
    ("sr", "serbian"),
    ("az", "azerbaijani"),
    ("sl", "slovenian"),
    ("kn", "kannada"),
    ("et", "estonian"),
    ("mk", "macedonian"),
    ("br", "breton"),
    ("eu", "basque"),
    ("is", "icelandic"),
    ("hy", "armenian"),
    ("ne", "nepali"),
    ("mn", "mongolian"),
    ("bs", "bosnian"),
    ("kk", "kazakh"),
    ("sq", "albanian"),
    ("sw", "swahili"),
    ("gl", "galician"),
    ("mr", "marathi"),
    ("pa", "punjabi"),
    ("si", "sinhala"),
    ("km", "khmer"),
    ("sn", "shona"),
    ("yo", "yoruba"),
    ("so", "somali"),
    ("af", "afrikaans"),
    ("oc", "occitan"),
    ("ka", "georgian"),
    ("be", "belarusian"),
    ("tg", "tajik"),
    ("sd", "sindhi"),
    ("gu", "gujarati"),
    ("am", "amharic"),
    ("yi", "yiddish"),
    ("lo", "lao"),
    ("uz", "uzbek"),
    ("fo", "faroese"),
    ("ht", "haitian creole"),
    ("ps", "pashto"),
    ("tk", "turkmen"),
    ("nn", "nynorsk"),
    ("mt", "maltese"),
    ("sa", "sanskrit"),
    ("lb", "luxembourgish"),
    ("my", "myanmar"),
    ("bo", "tibetan"),
    ("tl", "tagalog"),
    ("mg", "malagasy"),
    ("as", "assamese"),
    ("tt", "tatar"),
    ("haw", "hawaiian"),
    ("ln", "lingala"),
    ("ha", "hausa"),
    ("ba", "bashkir"),
    ("jw", "javanese"),
    ("su", "sundanese"),
];

pub fn detect_language(model: &Whisper, tokenizer: &Tokenizer, mel: &Tensor) -> Result<u32> {
    let device = mel.device();

    // TODO: Change this to use &str instead of heap strings.
    let language_token_ids = LANGUAGES
        .iter()
        .map(|(t, _)| token_id(tokenizer, &format!("<|{t}|>")))
        .collect::<Result<Vec<_>>>()?;

    let sot_token = token_id(tokenizer, constants::SOT_TOKEN)?;
    let (_, _, content_frames) = mel.dims3()?;
    let segment_size = usize::min(content_frames, constants::N_FRAMES);
    let mel_segment = mel.narrow(2, 0, segment_size)?;
    let audio_features = model.encoder.forward(&mel_segment)?;

    let tokens = Tensor::new(&[[sot_token]], device)?;
    let language_token_ids = Tensor::new(language_token_ids.as_slice(), device)?;

    let logits = model
        .decoder
        .forward(&tokens, &audio_features)?
        .i(0)?
        .i(0)?;

    let logits = logits.index_select(&language_token_ids, 0)?;
    let probs = candle_nn::ops::softmax(&logits, D::Minus1)?;
    let probs = probs.to_vec1::<f32>()?;
    let mut probs = LANGUAGES.iter().zip(probs.iter()).collect::<Vec<_>>();
    probs.sort_by(|(_, p1), (_, p2)| p2.total_cmp(p1));

    for ((_, language), p) in probs.iter().take(3) {
        println!("{language}: {p}")
    }

    let language = token_id(tokenizer, &format!("<|{}|>", probs[0].0 .0))?;
    Ok(language)
}
