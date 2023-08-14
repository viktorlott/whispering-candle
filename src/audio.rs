// Audio processing code, adapted from whisper.cpp
// https://github.com/ggerganov/whisper.cpp
use super::constants;

// Custom FFT using NEON intrinsics
mod neon {
    use core::arch::aarch64::*;

    const MAX_TERMS: usize = 10;

    unsafe fn vcosq_f32(values: float32x4_t) -> float32x4_t {
        let mut result = vdupq_n_f32(1.0);
        let mut x_power = vdupq_n_f32(1.0); // Start with x^0
        let mut factorial = 1.0;
        let mut sign = -1.0;

        for i in 1..=MAX_TERMS {
            x_power = vmulq_f32(x_power, values);
            factorial *= (2 * i) as f32 * (2 * i - 1) as f32;
            let term = vdivq_f32(vmulq_f32(x_power, x_power), vdupq_n_f32(factorial));
            result = vaddq_f32(result, vmulq_f32(vdupq_n_f32(sign), term));
            sign = -sign;
        }

        result
    }

    unsafe fn vsinq_f32(values: float32x4_t) -> float32x4_t {
        let mut result = values.clone();
        let mut x_power = values.clone(); // Start with x
        let mut factorial = 1.0;
        let mut sign = -1.0;

        for i in 1..=MAX_TERMS {
            x_power = vmulq_f32(x_power, vmulq_f32(values, values));
            factorial *= (2 * i + 1) as f32 * (2 * i) as f32;
            let term = vdivq_f32(x_power, vdupq_n_f32(factorial));
            result = vaddq_f32(result, vmulq_f32(vdupq_n_f32(sign), term));
            sign = -sign;
        }

        result
    }

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

        let mut out = vec![zero; n * 2];
        let mut even = Vec::with_capacity(n / 2);
        let mut odd = Vec::with_capacity(n / 2);

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
        let two_pi = 2.0 * std::f32::consts::PI;
        let n_f32 = n as f32;

        let mut k = 0;
        while k < n / 2 {
            let end = std::cmp::min(k + 4, n / 2);
            let len = end - k;

            // Prepare an array for k_values and fill it
            let mut k_array = [0.0f32; 4];
            for i in 0..len {
                k_array[i] = (k + i) as f32;
            }

            // Load k_values into a NEON vector
            let k_values = vld1q_f32(k_array.as_ptr());

            // Compute theta_values using vectorized operations
            let two_pi_vec = vdupq_n_f32(two_pi);
            let n_f32_vec = vdupq_n_f32(n_f32);
            let theta = vdivq_f32(vmulq_f32(two_pi_vec, k_values), n_f32_vec);

            let re = norm(vcosq_f32(theta));
            let im = norm(vnegq_f32(vsinq_f32(theta)));

            for i in 0..len {
                let re_odd = odd_fft[2 * (k + i)];
                let im_odd = odd_fft[2 * (k + i) + 1];

                out[2 * (k + i)] = even_fft[2 * (k + i)] + re[i] * re_odd - im[i] * im_odd;
                out[2 * (k + i) + 1] = even_fft[2 * (k + i) + 1] + re[i] * im_odd + im[i] * re_odd;

                out[2 * (k + i + n / 2)] = even_fft[2 * (k + i)] - re[i] * re_odd + im[i] * im_odd;
                out[2 * (k + i + n / 2) + 1] =
                    even_fft[2 * (k + i) + 1] - re[i] * im_odd - im[i] * re_odd;
            }

            k += len;
        }
        out
    }

    pub unsafe fn dft(inp: &[f32]) -> Vec<f32> {
        let n = inp.len();
        let two_pi = 2.0 * std::f32::consts::PI;
        let mut out = Vec::with_capacity(2 * n);
        let n_inv = vdupq_n_f32(1.0 / n as f32);

        for k in 0..n {
            let k_f32 = k as f32;
            let mut re = [0.0f32; 4];
            let mut im = [0.0f32; 4];

            for j in (0..n).step_by(4) {
                let j_vec = vaddq_f32(
                    vdupq_n_f32(j as f32),
                    vld1q_f32([0.0, 1.0, 2.0, 3.0].as_ptr()),
                );

                let two_pi_k = vdupq_n_f32(two_pi * k_f32);
                let theta = vmulq_f32(vmulq_f32(two_pi_k, j_vec), n_inv);

                let inp_values = vld1q_f32(inp[j..j + 4].as_ptr());
                let re_part = vmulq_f32(inp_values, vcosq_f32(theta));
                let im_part = vmulq_f32(inp_values, vsinq_f32(theta));

                let mut re_temp = [0.0f32; 4];
                let mut im_temp = [0.0f32; 4];
                vst1q_f32(re_temp.as_mut_ptr(), re_part);
                vst1q_f32(im_temp.as_mut_ptr(), im_part);

                for i in 0..4 {
                    re[i] += re_temp[i];
                    im[i] -= im_temp[i];
                }
            }

            out.push(re.iter().sum());
            out.push(im.iter().sum());
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
        unsafe {
            let c: Vec<f32> = fft(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0]);
            // NEON
            // #1 [9.0, 0.0, 0.70710677, 0.70710665, 6.4281416e-8, 0.99999994, -0.7071066, 0.7071068, -1.0, 0.0, -0.70710677, -0.70710665, -6.4281416e-8, -0.99999994, 0.7071066, -0.7071068]
            // #3 [9.0, 0.0, 0.70710677, 0.70710665, 6.4281416e-8, 0.99999994, -0.7071066, 0.7071068, -1.0, 0.0, -0.70710677, -0.70710665, -6.4281416e-8, -0.99999994, 0.7071066, -0.7071068]
            // #5 [9.0, 0.0, 0.70710677, 0.70710665, 6.4281416e-8, 0.99999994, -0.7071066, 0.7071068, -1.0, 0.0, -0.70710677, -0.70710665, -6.4281416e-8, -0.99999994, 0.7071066, -0.7071068]
            // #2 [9.0, 0.0, 0.7071069,  0.7071067,  4.371139e-8,  1.0,        -0.7071067, 0.7071068, -1.0, 0.0, -0.7071069,  -0.7071067,  -4.371139e-8,  -1.0,        0.7071067, -0.7071068]
            println!("{c:?}");
        }
    }
}
