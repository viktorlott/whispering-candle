use anyhow::{Error as E, Result};
use candle::{Device, Tensor};
use candle_nn::ops::softmax;

use rand::{distributions::Distribution, SeedableRng};
use tokenizers::Tokenizer;

use super::constants::*;
use super::model::Whisper;
use super::utils;

// Structure representing the result of decoding a segment of audio.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct DecodingResult {
    tokens: Vec<u32>,       // The tokens decoded from the audio segment.
    text: String,           // The text representation of the tokens.
    avg_logprob: f64,       // Average log probability for the decoded tokens.
    no_speech_prob: f64,    // Probability that the segment contains no speech.
    temperature: f64,       // Temperature value used during decoding.
    compression_ratio: f64, // Compression ratio for the decoded segment.
}

// Represents an audio segment with a start time, duration, and its decoding result.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Segment {
    start: f64,         // Start time of the segment.
    duration: f64,      // Duration of the segment.
    dr: DecodingResult, // The result of decoding this segment.
}

// Decoder structure containing the Whisper model, random number generator,
// tokenizer and tokens to suppress.
pub struct Decoder {
    model: Whisper,          // The Whisper model for decoding.
    rng: rand::rngs::StdRng, // Random number generator for stochastic decoding.
    tokenizer: Tokenizer,    // Tokenizer for converting tokens to text.
    suppress_tokens: Tensor, // Tensor of tokens to suppress.
    sot_token: u32,
    transcribe_token: u32,
    eot_token: u32,
    no_speech_token: u32,
    language_token: Option<u32>,
}

impl Decoder {
    pub fn new(
        model: Whisper,
        tokenizer: Tokenizer,
        seed: u64,
        device: &Device,
        language_token: Option<u32>,
    ) -> Result<Self> {
        // Create a tensor that indicates which tokens to suppress.
        let suppress_tokens: Vec<f32> = (0..model.config.vocab_size as u32)
            .map(|i| {
                if model.config.suppress_tokens.contains(&i) {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), device)?;
        let sot_token = utils::token_id(&tokenizer, SOT_TOKEN)?;
        let transcribe_token = utils::token_id(&tokenizer, TRANSCRIBE_TOKEN)?;
        let eot_token = utils::token_id(&tokenizer, EOT_TOKEN)?;
        let no_speech_token = utils::token_id(&tokenizer, NO_SPEECH_TOKEN)?;

        Ok(Self {
            model,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            tokenizer,
            suppress_tokens,
            sot_token,
            transcribe_token,
            eot_token,
            no_speech_token,
            language_token,
        })
    }

    // Function to decode a given Tensor (mel) using a specified temperature.
    pub fn decode(&mut self, mel: &Tensor, t: f64) -> Result<DecodingResult> {
        let model = &self.model;
        let audio_features = model.encoder.forward(mel)?;

        // println!("audio features: {:?}", audio_features.dims());

        let sample_len = model.config.max_target_positions / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;
        let mut tokens = vec![self.sot_token];
        if let Some(language_token) = self.language_token {
            tokens.push(language_token)
        }
        tokens.push(self.transcribe_token);
        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?;

            // The model expects a batch dim but this inference loop does not handle
            // it so we add it at this point.
            let tokens_t = tokens_t.unsqueeze(0)?;
            let logits = model.decoder.forward(&tokens_t, &audio_features)?;
            let logits = logits.squeeze(0)?;

            // Extract the no speech probability on the first iteration by looking at the first
            // token logits and the probability for the according token.
            if i == 0 {
                no_speech_prob = softmax(&logits.get(0)?, 0)?
                    .get(self.no_speech_token as usize)?
                    .to_scalar::<f32>()? as f64;
            }

            let (seq_len, _) = logits.dims2()?;
            let logits = logits
                .get(seq_len - 1)?
                .broadcast_add(&self.suppress_tokens)?;
            let next_token = if t > 0f64 {
                let prs = softmax(&(&logits / t)?, 0)?;
                let logits_v: Vec<f32> = prs.to_vec1()?;
                let distr = rand::distributions::WeightedIndex::new(&logits_v)?;
                distr.sample(&mut self.rng) as u32
            } else {
                let logits_v: Vec<f32> = logits.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };
            tokens.push(next_token);
            let prob = softmax(&logits, candle::D::Minus1)?
                .get(next_token as usize)?
                .to_scalar::<f32>()? as f64;
            if next_token == self.eot_token || tokens.len() > model.config.max_target_positions {
                break;
            }
            sum_logprob += prob.ln();
        }
        let text = self
            .tokenizer
            .decode(tokens.clone(), true)
            .map_err(E::msg)?;
        let avg_logprob = sum_logprob / tokens.len() as f64;

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature: t,
            compression_ratio: f64::NAN,
        })
    }

    // Attempts to decode a segment, falling back to different temperatures if needed.
    pub fn decode_with_fallback(&mut self, segment: &Tensor) -> Result<DecodingResult> {
        for (i, &t) in TEMPERATURES.iter().enumerate() {
            let dr: Result<DecodingResult> = self.decode(segment, t);
            if i == TEMPERATURES.len() - 1 {
                return dr;
            }
            // On errors, we try again with a different temperature.
            match dr {
                Ok(dr) => {
                    let needs_fallback = dr.compression_ratio > COMPRESSION_RATIO_THRESHOLD
                        || dr.avg_logprob < LOGPROB_THRESHOLD;
                    if !needs_fallback || dr.no_speech_prob > NO_SPEECH_THRESHOLD {
                        return Ok(dr);
                    }
                }
                Err(err) => {
                    println!("Error running at {t}: {err}")
                }
            }
        }

        // The loop above iterates through all the temperatures in `TEMPERATURES`.
        // For each temperature:
        // - If it's the last temperature, the function will return the result of the `decode`
        //   method, regardless of its outcome.
        // - If the decode attempt is successful and meets certain conditions, the function will
        //   return the `DecodingResult`.
        // - If the decode attempt results in an error, the function simply logs the error and
        //   continues to the next iteration.
        //
        // Given these paths, by the end of the loop, the function would have returned in all expected scenarios.
        // Therefore, the following line should never be reached in normal execution. If it is reached,
        // it indicates a logic bug or an unexpected scenario (like an empty `TEMPERATURES` list).
        unreachable!()
    }

    // Main function to decode the entire Tensor (mel) into segments and returns them.
    pub fn run(&mut self, mel: &Tensor) -> Result<Vec<Segment>> {
        let (_, _, content_frames) = mel.dims3()?;
        println!("Content frames: {content_frames}");

        let mut seek = 0;

        let max_segments: usize = (content_frames / N_FRAMES) + 1;
        let mut segments = Vec::with_capacity(max_segments);
        println!("Max segments: {max_segments}");

        while seek < content_frames {
            let start = std::time::Instant::now();
            let time_offset = (seek * HOP_LENGTH) as f64 / SAMPLE_RATE as f64;
            let segment_size = usize::min(content_frames - seek, N_FRAMES);
            println!("Segment size: {segment_size}");
            let mel_segment = mel.narrow(2, seek, segment_size)?;
            let segment_duration = (segment_size * HOP_LENGTH) as f64 / SAMPLE_RATE as f64;
            let dr = self.decode_with_fallback(&mel_segment)?;
            seek += segment_size;

            if dr.no_speech_prob > NO_SPEECH_THRESHOLD && dr.avg_logprob < LOGPROB_THRESHOLD {
                println!("no speech detected, skipping {seek} {dr:?}");
                continue;
            }
            let segment = Segment {
                start: time_offset,
                duration: segment_duration,
                dr,
            };

            println!("{seek}:{}", segment.dr.text);
            println!("{:?}", start.elapsed());
            segments.push(segment)
        }
        Ok(segments)
    }
}
