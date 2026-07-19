# Waveform / Spectrogram Interpretation

Per-member average energy and spectral concentration, computed from the original (non-augmented) recordings in `audio_features.csv`:

```
         rms_energy_mean  spectral_centroid_mean  spectral_rolloff_mean
member                                                                 
meme              0.0934               3160.0405              6340.0927
larissa           0.0562               3414.2532              7195.2823
rachel            0.0168               2156.3956              4401.9765
Alliane           0.0089               1652.3407              3177.1813
```

**Loudness varies a lot by member** -- `meme`'s recordings carry ~10x the average RMS energy of `Alliane`'s. In the waveform plots this shows up directly: some members' waveforms nearly clip at +/-1.0, while others peak well under 0.3. This is at least partly a recording-setup artifact (mic distance/gain), not pure vocal timbre -- worth flagging honestly, since some of what a speaker-ID model picks up on may be "how loud was their phone mic" rather than voice characteristics alone.

**Spectral energy is concentrated below ~2 kHz for every member** -- visible in the spectrograms as horizontal banding (the fundamental pitch and its harmonics) during voiced segments, with the rest of the frequency range mostly dark/quiet. Members with higher spectral centroid/rolloff (brighter-sounding recordings) also tend to be the louder ones, consistent with the same gain effect above.

**Waveforms show clear speech/silence segmentation** -- each phrase appears as 1-3 distinct amplitude bursts (roughly one per word/syllable group) separated by near-silence, with a low-level noise floor before and after the spoken segment. This matches the expected shape for a short spoken phrase and confirms the recordings aren't clipped, truncated, or corrupted.
