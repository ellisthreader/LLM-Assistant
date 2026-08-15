pub mod capture;
pub mod playback;
pub mod vad;

use std::io::Cursor;

/// Encode 16 kHz mono i16 samples as a WAV file in memory.
pub fn wav_from_samples(samples: &[i16]) -> Vec<u8> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: capture::TARGET_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec).expect("wav writer");
        for &s in samples {
            writer.write_sample(s).expect("wav sample");
        }
        writer.finalize().expect("wav finalize");
    }
    cursor.into_inner()
}
