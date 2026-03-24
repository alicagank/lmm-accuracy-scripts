# Evaluating the Accuracy of Speech-to-Text Technologies in Turkish-Accented English

This repository contains the scripts used in our study with Nuran Orhan on the performance of automatic speech recognition (ASR) systems on Turkish-accented English.

The project was presented at the *19th Student Conference of Linguistics* at İstanbul University (3 April 2026).

---

## Overview

This pipeline evaluates ASR output of Whipser's base model against a fixed reference passage (the “Stella” passage) taken from Speech Accent Archive (https://accent.gmu.edu/).

It consists of three main stages:

1. **Transcription** → Convert audio files into text using Whisper
2. **Evaluation** → Compare transcripts to the reference and compute error metrics
3. **Analysis** → Model and visualise error patterns across speakers

For more information please refer to the presentation on:
