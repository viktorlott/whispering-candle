// Audio processing code, adapted from whisper.cpp
// https://github.com/ggerganov/whisper.cpp
use super::constants;

const TWO_PI: f32 = 2.0 * std::f32::consts::PI;
const MAX_TERMS: usize = 7;
const COSINE_FACTORIAL_SERIES: [f32; MAX_TERMS] = super::utils::factorial_sequence(false);
const SINE_FACTORIAL_SERIES: [f32; MAX_TERMS] = super::utils::factorial_sequence(true);

// Custom FFT using NEON intrinsics
mod neon {
    use super::*;
    use core::arch::aarch64::*;

    // https://en.wikipedia.org/wiki/Taylor_series
    #[inline(always)]
    unsafe fn vcosq_f32(values: float32x4_t) -> float32x4_t {
        let mut result = vdupq_n_f32(1.0);
        let mut x_power = vdupq_n_f32(1.0); // Start with x^0
        let mut sign = -1.0;

        // Taylor series for cosine:
        //      1 *  (2 * 1 - 1 = 1)    *    (2 * 1 = 2) = 2      | 2!
        //      2 *  (2 * 2 - 1 = 3)    *    (2 * 2 = 4) = 24     | 4!
        //     24 *  (2 * 3 - 1 = 5)    *    (2 * 3 = 6) = 720    | 6!
        for i in 1..=MAX_TERMS {
            // x^2, x^4, x^6, ...
            x_power = vmulq_f32(x_power, values);
            // 2!, 4!, 6!, ...
            let term = vdivq_f32(
                vmulq_f32(x_power, x_power),
                vdupq_n_f32(COSINE_FACTORIAL_SERIES[i - 1]),
            );
            // x^2/2!, x^4/4!, ...
            result = vaddq_f32(result, vmulq_f32(vdupq_n_f32(sign), term));
            sign = -sign;
        }

        result
    }

    // https://en.wikipedia.org/wiki/Taylor_series
    #[inline(always)]
    unsafe fn vsinq_f32(values: float32x4_t) -> float32x4_t {
        let mut result = values;
        let mut x_power = values;
        let mut sign = -1.0;

        // Taylor series for cosine:
        //      1 *  (2 * 1 + 1 = 3)    *    (2 * 1 = 2) = 6      | 3!
        //      6 *  (2 * 2 + 1 = 5)    *    (2 * 2 = 4) = 120    | 5!
        //    120 *  (2 * 3 + 1 = 7)    *    (2 * 3 = 6) = 5040   | 7!
        for i in 1..=MAX_TERMS {
            // x^3, x^5, x^7, ...
            x_power = vmulq_f32(x_power, vmulq_f32(values, values));
            // 3!, 5!, 7!, ...
            let term = vdivq_f32(x_power, vdupq_n_f32(SINE_FACTORIAL_SERIES[i - 1]));
            // x^3/3!, x^5/5!, ...
            result = vaddq_f32(result, vmulq_f32(vdupq_n_f32(sign), term));
            sign = -sign;
        }

        result
    }

    #[inline(always)]
    unsafe fn norm(values: float32x4_t) -> [f32; 4] {
        let mut output = [0.0f32; 4];
        vst1q_f32(output.as_mut_ptr(), values);
        output
    }

    pub unsafe fn fft(inp: &[f32]) -> Vec<f32> {
        let n = inp.len();
        let zero = 0.0f32;

        // Base cases for recursion.
        if n == 1 {
            return vec![inp[0], zero];
        }
        if n % 2 == 1 {
            return dft(inp);
        }

        let half_n = n / 2;

        let mut out = vec![zero; n * 2];
        let mut even = Vec::with_capacity(half_n);
        let mut odd = Vec::with_capacity(half_n);

        // Split the input into even and odd components.
        for (i, &inp_val) in inp.iter().enumerate() {
            if i % 2 == 0 {
                even.push(inp_val)
            } else {
                odd.push(inp_val);
            }
        }

        // Recursive FFT calls for even and odd parts.
        let even_fft = fft(&even);
        let odd_fft = fft(&odd);

        // Combine the FFT results from the even and odd parts.
        let two_pi_vec = vdupq_n_f32(TWO_PI);
        let n_f32_vec = vdupq_n_f32(n as f32);

        let mut k = 0;
        while k < half_n {
            // end should be in between the [4, half_n] interval
            let end = std::cmp::min(k + 4, half_n);
            let len = end - k;

            // Prepare an array for k_values and fill it
            let k_base = vdupq_n_f32(k as f32);
            let increment = vld1q_f32([0.0, 1.0, 2.0, 3.0].as_ptr());
            let k_values = vaddq_f32(k_base, increment);

            // Compute theta_values using vectorized operations
            let theta = vdivq_f32(vmulq_f32(two_pi_vec, k_values), n_f32_vec);

            let re = norm(vcosq_f32(theta));
            let im = norm(vnegq_f32(vsinq_f32(theta)));

            for i in 0..len {
                let offset = k + i;
                let left_current = 2 * offset;
                let left_next = left_current + 1;

                let re_odd = odd_fft[left_current];
                let im_odd = odd_fft[left_next];

                let im_time_im_odd = im[i] * im_odd;
                let im_time_re_odd = im[i] * re_odd;

                let re_time_re_odd = re[i] * re_odd;
                let re_time_im_odd = re[i] * im_odd;

                out[left_current] = even_fft[left_current] + re_time_re_odd - im_time_im_odd;
                out[left_next] = even_fft[left_next] + re_time_im_odd + im_time_re_odd;

                let right_current = 2 * (offset + half_n);
                let right_next = right_current + 1;

                out[right_current] = even_fft[left_current] - re_time_re_odd + im_time_im_odd;
                out[right_next] = even_fft[left_next] - re_time_im_odd - im_time_re_odd;
            }

            k += len;
        }
        out
    }

    pub unsafe fn dft(inp: &[f32]) -> Vec<f32> {
        let n = inp.len();
        let mut out = Vec::with_capacity(2 * n);
        let n_inv = vdupq_n_f32(1.0 / n as f32);

        for k in 0..n {
            let two_pi_k = vdupq_n_f32(k as f32 * TWO_PI);
            let mut re_vec = vdupq_n_f32(0.0);
            let mut im_vec = vdupq_n_f32(0.0);

            for j in (0..n).step_by(4) {
                let j_vec = vaddq_f32(
                    vdupq_n_f32(j as f32),
                    vld1q_f32([0.0, 1.0, 2.0, 3.0].as_ptr()),
                );

                let theta = vmulq_f32(vmulq_f32(two_pi_k, j_vec), n_inv);
                let inp_values = vld1q_f32(inp[j..j + 4].as_ptr());
                let re_part = vmulq_f32(inp_values, vcosq_f32(theta));
                let im_part = vmulq_f32(inp_values, vsinq_f32(theta));

                // Directly accumulate into re_vec and im_vec
                re_vec = vaddq_f32(re_vec, re_part);
                im_vec = vsubq_f32(im_vec, im_part);
            }

            // Summing using NEON intrinsics
            out.push(vaddvq_f32(re_vec));
            out.push(vaddvq_f32(im_vec));
        }
        out
    }
}

// Define a trait for floating point numbers that encompasses various traits from the `num_traits` crate.
pub trait Float: num_traits::Float + num_traits::FloatConst + num_traits::NumAssign {}

// Implement the `Float` trait for 32-bit and 64-bit floating point numbers.
impl Float for f32 {}
impl Float for f64 {}

// https://github.com/ggerganov/whisper.cpp/blob/4774d2feb01a772a15de81ffc34b34a1f294f020/whisper.cpp#L2357
// Implements the Fast Fourier Transform (FFT) using the Cooley-Tukey algorithm. The FFT is a way to
// transform a signal from its original domain (often time) into the frequency domain. This function
// recursively breaks down the input until it reaches base cases, then it combines results to
// produce the FFT of the entire signal.
fn fft<T: Float>(inp: &[T]) -> Vec<T> {
    let n = inp.len();
    let zero = T::zero();

    // Base cases for recursion.
    if n == 1 {
        return vec![inp[0], zero];
    }
    if n % 2 == 1 {
        return dft(inp);
    }

    let mut out = vec![zero; n * 2];
    let mut even = Vec::with_capacity(n / 2);
    let mut odd = Vec::with_capacity(n / 2);

    // Split the input into even and odd components.
    for (i, &inp) in inp.iter().enumerate() {
        if i % 2 == 0 {
            even.push(inp)
        } else {
            odd.push(inp);
        }
    }

    // Recursive FFT calls for even and odd parts.
    let even_fft = fft(&even);
    let odd_fft = fft(&odd);

    // Combine the FFT results from the even and odd parts.
    let two_pi = T::PI() + T::PI();
    let n_t = T::from(n).unwrap();
    for k in 0..n / 2 {
        let k_t = T::from(k).unwrap();
        let theta = two_pi * k_t / n_t;
        let re = theta.cos();
        let im = -theta.sin();

        let re_odd = odd_fft[2 * k];
        let im_odd = odd_fft[2 * k + 1];

        out[2 * k] = even_fft[2 * k] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;

        out[2 * (k + n / 2)] = even_fft[2 * k] - re * re_odd + im * im_odd;
        out[2 * (k + n / 2) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
    out
}

// Direct Fourier Transform (DFT) computes the Discrete Fourier Transform directly without any
// optimization. The DFT represents a signal in the frequency domain. It's computationally
// expensive, especially for large inputs, which is why FFT algorithms like Cooley-Tukey are often
// preferred.
fn dft<T: Float>(inp: &[T]) -> Vec<T> {
    let zero = T::zero();
    let n = inp.len();
    let two_pi = T::PI() + T::PI();

    let mut out = Vec::new();
    out.reserve(2 * n);
    let n_t = T::from(n).unwrap();

    // Calculate the DFT directly.
    for k in 0..n {
        let k_t = T::from(k).unwrap();
        let mut re = zero;
        let mut im = zero;

        for (j, &inp) in inp.iter().enumerate() {
            let j_t = T::from(j).unwrap();
            let angle = two_pi * k_t * j_t / n_t;
            re += inp * angle.cos();
            im -= inp * angle.sin();
        }

        out.push(re);
        out.push(im);
    }
    out
}

/// This function computes a mel spectrogram for a section of an audio signal. Mel spectrograms are
/// representations of the short-term power spectrum of sound, emphasizing aspects that human ears
/// are sensitive to. This version is designed to be work-stealing, meaning it's intended to be part
/// of a parallel processing system.
///
/// https://github.com/ggerganov/whisper.cpp/blob/4774d2feb01a772a15de81ffc34b34a1f294f020/whisper.cpp#L2414
///
/// # Arguments:
///
/// * `ith`: The index of the current processing window or batch. Often used when processing in
///   parallel or batches.
///
/// * `hann`: A pre-computed Hann window. The Hann window is a type of tapering window used in FFT
///   to minimize the discontinuity at the beginning and end of the sampled window.
///
/// * `samples`: The raw audio samples to process.
///
/// * `filters`: Mel filter banks to convert FFT bins to Mel scale. These are triangular filters
///   used to simulate the human ear's frequency response.
///
/// * `fft_size`: The size of the FFT window, i.e., how many samples are included in each FFT
///   computation. This determines the frequency resolution of the resulting spectrogram.
///
/// * `fft_step`: The step size between successive FFT windows. This determines how much the window
///   is moved for each FFT computation and thus affects time resolution.
///
/// * `speed_up`: A flag that indicates if the function should use any optimizations to speed up the
///   computation.
///
/// * `n_len`: The length of the processed audio segment, often used for ensuring buffer sizes or
///   memory allocations.
///
/// * `n_mel`: The number of Mel bands or Mel filters to use when converting the spectrogram.
///   Determines the resolution on the Mel scale.
///
/// * `n_threads`: The number of threads to use for parallel processing, if applicable.
///
#[allow(clippy::too_many_arguments)]
fn log_mel_spectrogram_w<T: Float>(
    ith: usize,
    hann: &[T],
    samples: &[T],
    filters: &[T],
    fft_size: usize,
    fft_step: usize,
    speed_up: bool,
    n_len: usize,
    n_mel: usize,
    n_threads: usize,
) -> Vec<T> {
    let n_fft = if speed_up {
        1 + fft_size / 4
    } else {
        1 + fft_size / 2
    };

    let zero = T::zero();
    let half = T::from(0.5).unwrap();
    let mut fft_in = vec![zero; fft_size];
    let mut mel = vec![zero; n_len * n_mel];

    for i in (ith..n_len).step_by(n_threads) {
        let offset = i * fft_step;

        // apply Hanning window
        for j in 0..fft_size {
            fft_in[j] = if offset + j < samples.len() {
                hann[j] * samples[offset + j]
            } else {
                zero
            }
        }

        // FFT -> mag^2
        let mut fft_out: Vec<T> = fft(&fft_in);

        for j in 0..fft_size {
            fft_out[j] = fft_out[2 * j] * fft_out[2 * j] + fft_out[2 * j + 1] * fft_out[2 * j + 1];
        }
        for j in 1..fft_size / 2 {
            let v = fft_out[fft_size - j];
            fft_out[j] += v;
        }

        if speed_up {
            // scale down in the frequency domain results in a speed up in the time domain
            for j in 0..n_fft {
                fft_out[j] = half * (fft_out[2 * j] + fft_out[2 * j + 1]);
            }
        }

        // mel spectrogram
        for j in 0..n_mel {
            let mut sum = zero;
            for k in 0..n_fft {
                sum += fft_out[k] * filters[j * n_fft + k];
            }
            mel[j * n_len + i] = T::max(sum, T::from(1e-10).unwrap()).log10();
        }
    }
    mel
}

/// Computes the mel spectrogram of an audio signal.
/// It uses the FFT and mel filter banks to generate a representation that models human ear perception.
/// The function also involves windowing (with the Hann window) the input signal
/// and then scaling the output to better fit into a desired dynamic range.
///
/// # Arguments:
/// * `samples`: The raw audio samples to process.
/// * `filters`: Mel filter banks to convert FFT bins to Mel scale.
/// * `fft_size`: The size of the FFT window, affecting frequency resolution.
/// * `fft_step`: The step size between successive FFT windows, affecting time resolution.
/// * `n_mel`: The number of Mel bands or Mel filters to use.
/// * `speed_up`: Flag to indicate if optimizations for speed are to be used.
fn log_mel_spectrogram_<T: Float + std::fmt::Display>(
    samples: &[T],
    filters: &[T],
    fft_size: usize,
    fft_step: usize,
    n_mel: usize,
    speed_up: bool,
) -> Vec<T> {
    let zero = T::zero();
    let two_pi = T::PI() + T::PI();
    let half = T::from(0.5).unwrap();
    let one = T::from(1.0).unwrap();
    let four = T::from(4.0).unwrap();
    let fft_size_t = T::from(fft_size).unwrap();

    let hann: Vec<T> = (0..fft_size)
        .map(|i| half * (one - ((two_pi * T::from(i).unwrap()) / fft_size_t).cos()))
        .collect();
    let n_len = samples.len() / fft_step;

    // pad audio with at least one extra chunk of zeros
    let pad = 100 * constants::CHUNK_LENGTH / 2;
    let n_len = if n_len % pad != 0 {
        (n_len / pad + 1) * pad
    } else {
        n_len
    };
    let n_len = n_len + pad;
    let samples = {
        let mut samples_padded = samples.to_vec();
        let to_add = n_len * fft_step - samples.len();
        samples_padded.extend(std::iter::repeat(zero).take(to_add));
        samples_padded
    };

    // Use a single thread for now.
    let mut mel = log_mel_spectrogram_w(
        0, &hann, &samples, filters, fft_size, fft_step, speed_up, n_len, n_mel, 1,
    );

    let mmax = mel
        .iter()
        .max_by(|&u, &v| u.partial_cmp(v).unwrap_or(std::cmp::Ordering::Greater))
        .copied()
        .unwrap_or(zero)
        - T::from(8).unwrap();
    for m in mel.iter_mut() {
        let v = T::max(*m, mmax);
        *m = v / four + one
    }
    mel
}

// A public interface to transform raw Pulse-Code Modulation (PCM) audio samples into a mel spectrogram.
// PCM is a method used to digitally represent analog signals. This function will produce a mel spectrogram,
// which can then be used for various audio processing tasks, including speech recognition and music analysis.
pub fn pcm_to_mel<T: Float + std::fmt::Display>(
    samples: &[T],
    filters: &[T],
) -> anyhow::Result<Vec<T>> {
    Ok(log_mel_spectrogram_(
        samples,
        filters,
        constants::N_FFT,
        constants::HOP_LENGTH,
        constants::N_MELS,
        false,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        println!("{:?}", SINE_FACTORIAL_SERIES);
        println!("{:?}", COSINE_FACTORIAL_SERIES);
        unsafe {
            let mut y: Vec<f32> = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0].to_vec();

            let z = neon::fft(&mut y);
            // NEON
            // #1 [9.0, 0.0, 0.70710677, 0.70710665, 6.4281416e-8, 0.99999994, -0.7071066, 0.7071068, -1.0, 0.0, -0.70710677, -0.70710665, -6.4281416e-8, -0.99999994, 0.7071066, -0.7071068]
            // #3 [9.0, 0.0, 0.70710677, 0.70710665, 6.4281416e-8, 0.99999994, -0.7071066, 0.7071068, -1.0, 0.0, -0.70710677, -0.70710665, -6.4281416e-8, -0.99999994, 0.7071066, -0.7071068]
            // #5 [9.0, 0.0, 0.70710677, 0.70710665, 6.4281416e-8, 0.99999994, -0.7071066, 0.7071068, -1.0, 0.0, -0.70710677, -0.70710665, -6.4281416e-8, -0.99999994, 0.7071066, -0.7071068]
            //    [9.0, 0.0, 0.70710707, 0.70710635, 5.290476e-7,  0.9999999,  -0.7070955, 0.70716643, -1.0, 0.0, -0.70710707, -0.70710635, -5.290476e-7, -0.9999999, 0.7070955, -0.70716643]
            // #2 [9.0, 0.0, 0.7071069,  0.7071067,  4.371139e-8,  1.0,        -0.7071067, 0.7071068, -1.0, 0.0, -0.7071069,  -0.7071067,  -4.371139e-8,  -1.0,        0.7071067, -0.7071068]
            println!("{z:?}");
        }
    }
}
